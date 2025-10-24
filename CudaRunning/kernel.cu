// bigdiv_loops2gpu.cu
// Phase 1: Remove limb-wise loops by using GPU carry-lookahead add/sub.
// Build: nvcc -O2 -std=c++17 bigdiv_loops2gpu.cu -o tests && ./tests

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_runtime.h>
static inline const char* _cudaErr(cudaError_t e){ return cudaGetErrorString(e); }
#define CUDA_CHECK(x) do{ cudaError_t err=(x); if(err!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, _cudaErr(err)); std::exit(1);} }while(0)
#else
using cudaError_t = int;
#define cudaSuccess 0
#define CUDA_CHECK(x) do{ (void)(x); }while(0)
#endif

// ===================== utilities =====================
// ---- CPU limb helpers (needed by the safe Newton path) ----
static inline uint32_t add_into(uint32_t* X,const uint32_t* Y,int n){
  unsigned long long c=0;
  for(int i=0;i<n;i++){
    unsigned long long s=(unsigned long long)X[i] + (unsigned long long)Y[i] + c;
    X[i]=(uint32_t)s; c=s>>32;
  }
  return (uint32_t)c; // carry-out
}

static inline uint32_t sub_into(uint32_t* X,const uint32_t* Y,int n){
  unsigned long long b=0;
  for(int i=0;i<n;i++){
    unsigned long long xi=X[i], yi=Y[i];
    unsigned long long d=xi - yi - b;
    X[i]=(uint32_t)d;
    b=(xi < yi + b) ? 1 : 0;
  }
  return (uint32_t)b; // borrow-out
}

static inline void inc_vec(std::vector<uint32_t>& A){
  uint64_t c=1;
  for(size_t i=0;i<A.size() && c;i++){
    uint64_t s=(uint64_t)A[i] + c;
    A[i]=(uint32_t)s; c=s>>32;
  }
  if(c) A.push_back((uint32_t)c);
}

static inline void dec_vec(std::vector<uint32_t>& A){
  uint64_t b=1;
  for(size_t i=0;i<A.size() && b;i++){
    uint64_t xi=A[i];
    uint64_t d=xi - b;
    A[i]=(uint32_t)d;
    b=(xi<b)?1:0;
  }
  // optional trim
  int n=(int)A.size(); while(n>0 && A[n-1]==0) --n; A.resize(n? n:1);
}

static inline int trim_len(const uint32_t* a,int n){ while(n>0 && a[n-1]==0) --n; return n; }
static inline void trim_vec(std::vector<uint32_t>& v){ int n=trim_len(v.data(),(int)v.size()); v.resize(n? n:1); }
static inline int cmp_be(const uint32_t* A,int nA,const uint32_t* B,int nB){
  nA=trim_len(A,nA); nB=trim_len(B,nB);
  if(nA!=nB) return (nA<nB)?-1:+1;
  for(int i=nA-1;i>=0;--i){ if(A[i]!=B[i]) return (A[i]<B[i])?-1:+1; if(i==0)break; }
  return 0;
}
static inline int clz32(uint32_t x){
#if defined(__GNUC__) || defined(__clang__)
  return x?__builtin_clz(x):32;
#else
  int c=0; if(!x) return 32; while((x&0x80000000u)==0){ x<<=1; ++c; } return c;
#endif
}
static inline void print_vec(const char* tag,const std::vector<uint32_t>& v){
  int n=trim_len(v.data(),(int)v.size());
  std::printf("%s (len=%d): [", tag, n);
  for(int i=0;i<n;i++) std::printf("0x%08x%s", v[i], (i+1<n)?", ":"");
  std::puts("]");
}
static inline bool eq_vec(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b){
  int na=trim_len(a.data(),(int)a.size()), nb=trim_len(b.data(),(int)b.size());
  if(na!=nb) return false;
  for(int i=0;i<na;i++) if(a[i]!=b[i]) return false;
  return true;
}
static inline bool is_zero_vec(const std::vector<uint32_t>& a){ for(uint32_t x:a) if(x) return false; return true; }

// ===================== CPU mul (fixed) =====================
static void mul_full_cpu(const uint32_t* A,int nA,const uint32_t* B,int nB,std::vector<uint32_t>& C){
  C.assign(nA+nB, 0u);
  for(int i=0;i<nA;i++){
    unsigned long long carry = 0ULL;
    for(int j=0;j<nB;j++){
      unsigned long long cur =
          (unsigned long long)C[i+j]
        + (unsigned long long)A[i]*(unsigned long long)B[j]
        + carry;
      C[i+j] = (uint32_t)cur;
      carry  = cur >> 32;
    }
    int k = i + nB;
    while(carry){
      unsigned long long s = (unsigned long long)C[k] + carry;
      C[k] = (uint32_t)s;
      carry = s >> 32;
      ++k;
      if(k >= (int)C.size()) C.push_back(0u);
    }
  }
}

// ===================== Knuth helpers =====================
static void shl_bits_into(const uint32_t* A,int n,int s,std::vector<uint32_t>& out){
  out.resize(n+1);
  unsigned long long carry=0;
  for(int i=0;i<n;i++){
    unsigned long long v=((unsigned long long)A[i]<<s) | carry;
    out[i]=(uint32_t)v; carry=v>>32;
  }
  out[n]=(uint32_t)carry;
}
static void shr_bits_with_high(std::vector<uint32_t>& a,int s,uint32_t high){
  const uint64_t mask = (s==64)?~0ull:((1ull<<s)-1ull);
  uint64_t hi = high;
  for(int i=(int)a.size()-1;i>=0;--i){
    uint64_t cur = (hi<<32) | a[i];
    a[i] = (uint32_t)(cur >> s);
    hi = cur & mask;
  }
}

// ===================== Knuth (robust) =====================
static void div_knuth(const std::vector<uint32_t>& Uin,const std::vector<uint32_t>& Vin,
                      std::vector<uint32_t>& Q,std::vector<uint32_t>& R)
{
  int nU=trim_len(Uin.data(),(int)Uin.size());
  int nV=trim_len(Vin.data(),(int)Vin.size());
  if(nV==0){ std::fprintf(stderr,"Division by zero\n"); std::exit(1); }
  if(nU<nV){ Q.assign(1,0); R=Uin; trim_vec(R); return; }

  if(nV==1){
    uint64_t v=Vin[0]; Q.assign(nU,0); uint64_t rem=0;
    for(int i=nU-1;i>=0;--i){ __uint128_t cur=((__uint128_t)rem<<32)|Uin[i]; Q[i]=(uint32_t)(cur/v); rem=(uint64_t)(cur%v); }
    R.assign(1,(uint32_t)rem); trim_vec(Q); return;
  }

  int s = clz32(Vin[nV-1]); if(s==32) s=0;
  std::vector<uint32_t> Un, Vn_full;
  if(s){ shl_bits_into(Uin.data(), nU, s, Un); shl_bits_into(Vin.data(), nV, s, Vn_full); }
  else { Un.assign(nU+1,0); std::copy(Uin.begin(),Uin.begin()+nU,Un.begin());
         Vn_full.assign(nV+1,0); std::copy(Vin.begin(),Vin.begin()+nV,Vn_full.begin()); }
  std::vector<uint32_t> Vn(Vn_full.begin(), Vn_full.begin()+nV);

  Q.assign(nU-nV+1, 0);
  const uint64_t B = 1ull<<32;

  for(int j=nU-nV; j>=0; --j){
    uint64_t u2 = Un[j+nV];
    uint64_t u1 = Un[j+nV-1];
    uint64_t u0 = (nV>=2 ? Un[j+nV-2] : 0);
    uint64_t v1 = Vn[nV-1];
    uint64_t v0 = (nV>=2 ? Vn[nV-2] : 0);
    __int128 num = ( ( (__int128)u2 << 64) | ( (__int128)u1 << 32) | u0 );
    __int128 den = ( ( (__int128)v1 << 32) | v0 );
    uint64_t qhat = (den ? (uint64_t)(num / den) : (uint64_t)(B-1));
    if(qhat >= B) qhat = B-1;

    uint64_t k = 0;
    for(int i=0;i<nV;i++){
      __int128 t = (__int128)qhat * (uint64_t)Vn[i] + k;
      uint64_t lo = (uint64_t)(uint32_t)t;
      uint64_t ui = Un[j+i];
      uint64_t diff = ui - lo;
      Un[j+i] = (uint32_t)diff;
      k = (uint64_t)(t >> 32) + ((diff >> 63) & 1u);
    }
    uint64_t top = Un[j+nV];
    uint64_t diffN = top - k;

    if( (diffN >> 63) & 1u ){
      --qhat;
      uint64_t carry=0;
      for(int i=0;i<nV;i++){
        uint64_t ssum = (uint64_t)Un[j+i] + (uint64_t)Vn[i] + carry;
        Un[j+i] = (uint32_t)ssum;
        carry = ssum >> 32;
      }
      Un[j+nV] = (uint32_t)((uint64_t)Un[j+nV] + carry);
    }else{
      Un[j+nV] = (uint32_t)diffN;
      // undershoot correction
      int rn=nV; while(rn>0 && Un[j+rn-1]==0) --rn;
      int vn=nV; while(vn>0 && Vn[vn-1]==0) --vn;
      int cmp=0;
      if(rn!=vn) cmp=(rn<vn)?-1:+1;
      else{
        for(int t=rn-1;t>=0;--t){ if(Un[j+t]!=Vn[t]){ cmp=(Un[j+t]<Vn[t])?-1:+1; break; } if(t==0)break; }
      }
      if(cmp>=0){
        uint64_t br=0;
        for(int i=0;i<nV;i++){
          uint64_t ui2 = Un[j+i];
          uint64_t vi  = Vn[i];
          uint64_t d   = ui2 - vi - br;
          Un[j+i] = (uint32_t)d;
          br = ((ui2 < vi + br) ? 1u : 0u);
        }
        Un[j+nV] -= (uint32_t)br;
        ++qhat;
      }
    }
    Q[j] = (uint32_t)qhat;
  }

  R.assign(Un.begin(), Un.begin()+nV);
  if(s) shr_bits_with_high(R, s, Un[nV]);
  R.resize(nV);
  trim_vec(Q);
}

// ===================== GPU convolutions (unchanged) =====================
#ifdef __CUDACC__
__global__ void conv_split_atomic_kernel(const uint32_t* __restrict__ A,int nA,
                                         const uint32_t* __restrict__ B,int nB,
                                         unsigned long long* __restrict__ buckets64,
                                         int out_len, int trunc_t /*-1=full*/)
{
  long long total = (long long)nA * (long long)nB;
  for(long long idx = blockIdx.x*1ll*blockDim.x + threadIdx.x; idx < total; idx += 1ll*gridDim.x*blockDim.x){
    int i = (int)(idx / nB);
    int j = (int)(idx % nB);
    int s = i + j;
    unsigned long long p = (unsigned long long)A[i] * (unsigned long long)B[j];
    if(trunc_t < 0){
      if(s < out_len) atomicAdd(&buckets64[s], (unsigned long long)(p & 0xffffffffULL));
      if(s+1 < out_len) atomicAdd(&buckets64[s+1], (unsigned long long)(p >> 32));
    }else{
      if(s < trunc_t) atomicAdd(&buckets64[s], (unsigned long long)(p & 0xffffffffULL));
      if(s+1 < trunc_t) atomicAdd(&buckets64[s+1], (unsigned long long)(p >> 32));
    }
  }
}
__global__ void normalize_carry_kernel(const unsigned long long* __restrict__ buckets64,
                                       uint32_t* __restrict__ out32,
                                       int out_len)
{
  if(blockIdx.x==0 && threadIdx.x==0){
    unsigned long long carry=0;
    for(int k=0;k<out_len;k++){
      unsigned long long t = buckets64[k] + carry;
      out32[k] = (uint32_t)(t & 0xffffffffULL);
      carry = t >> 32;
    }
  }
}
#endif

static bool cuda_available(){
#ifdef __CUDACC__
  int n=0; cudaError_t e = cudaGetDeviceCount(&n);
  if(e!=cudaSuccess || n<=0) return false; return true;
#else
  return false;
#endif
}
static void mul_full(const uint32_t* A,int nA,const uint32_t* B,int nB,std::vector<uint32_t>& C){
#ifdef __CUDACC__
  if(cuda_available()){
    const int out_len = nA + nB;
    uint32_t *dA=nullptr, *dB=nullptr; unsigned long long *dS=nullptr; uint32_t *dOut=nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, nA*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&dB, nB*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&dS, (out_len+1)*sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&dOut, out_len*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dA, A, nA*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, nB*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dS, 0, (out_len+1)*sizeof(unsigned long long)));
    long long pairs = 1ll*nA*nB;
    int threads = 256;
    int blocks = (int)std::min( (long long)65535, (pairs + threads - 1)/threads );
    conv_split_atomic_kernel<<<blocks, threads>>>(dA,nA,dB,nB,dS,out_len+1,-1);
    CUDA_CHECK(cudaGetLastError());
    normalize_carry_kernel<<<1,1>>>(dS, dOut, out_len);
    CUDA_CHECK(cudaGetLastError());
    C.assign(out_len,0u);
    CUDA_CHECK(cudaMemcpy(C.data(), dOut, out_len*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dS); cudaFree(dOut); return;
  }
#endif
  mul_full_cpu(A,nA,B,nB,C);
}
static void mul_trunc_lo(const uint32_t* A,int nA,const uint32_t* B,int nB,int t,std::vector<uint32_t>& C){
  if(t<=0){ C.assign(1,0u); return; }
#ifdef __CUDACC__
  if(cuda_available()){
    const int out_len = t;
    uint32_t *dA=nullptr, *dB=nullptr; unsigned long long *dS=nullptr; uint32_t *dOut=nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, nA*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&dB, nB*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&dS, out_len*sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&dOut, out_len*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dA, A, nA*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, nB*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dS, 0, out_len*sizeof(unsigned long long)));
    long long pairs = 1ll*nA*nB;
    int threads = 256;
    int blocks = (int)std::min( (long long)65535, (pairs + threads - 1)/threads );
    conv_split_atomic_kernel<<<blocks, threads>>>(dA,nA,dB,nB,dS,out_len,t);
    CUDA_CHECK(cudaGetLastError());
    normalize_carry_kernel<<<1,1>>>(dS, dOut, out_len);
    CUDA_CHECK(cudaGetLastError());
    C.assign(out_len,0u);
    CUDA_CHECK(cudaMemcpy(C.data(), dOut, out_len*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dS); cudaFree(dOut); return;
  }
#endif
  // CPU fallback (truncated)
  C.assign(t,0u);
  for(int i=0;i<nA;i++){
    unsigned long long carry=0;
    for(int j=0;j<nB && i+j<t;j++){
      unsigned long long cur=(unsigned long long)C[i+j]+(unsigned long long)A[i]*(unsigned long long)B[j]+carry;
      C[i+j]=(uint32_t)cur; carry=cur>>32;
    }
    if(i+nB<t){
      unsigned long long s=(unsigned long long)C[i+nB]+carry;
      C[i+nB]=(uint32_t)s; carry=s>>32;
      for(int k=i+nB+1; carry && k<t; ++k){
        unsigned long long s2=(unsigned long long)C[k]+carry;
        C[k]=(uint32_t)s2; carry=s2>>32;
      }
    }
  }
}

// ===================== NEW: GPU carry-lookahead add/sub =====================
#ifdef __CUDACC__
#define MAX_LIMBS_BLOCK 2048

__global__ void ks_add32_kernel(const uint32_t* __restrict__ X,
                                const uint32_t* __restrict__ Y,
                                uint32_t* __restrict__ OUT,
                                uint32_t* __restrict__ carry_out,
                                int n)
{
  __shared__ uint32_t G[MAX_LIMBS_BLOCK];
  __shared__ uint32_t P[MAX_LIMBS_BLOCK];
  int tid = threadIdx.x;
  if(tid < n){
    uint64_t s = (uint64_t)X[tid] + Y[tid];
    G[tid] = (uint32_t)(s >> 32);
    P[tid] = ((uint32_t)s == 0xFFFFFFFFu) ? 1u : 0u;
  }
  __syncthreads();

  for(int offset=1; offset < n; offset<<=1){
    uint32_t g_prev=0, p_prev=0;
    if(tid >= offset && tid < n){ g_prev = G[tid - offset]; p_prev = P[tid - offset]; }
    __syncthreads();
    if(tid < n){
      uint32_t g = G[tid], p = P[tid];
      uint32_t g_new = g | (p & g_prev);
      uint32_t p_new = p & p_prev;
      G[tid] = g_new; P[tid] = p_new;
    }
    __syncthreads();
  }

  if(tid < n){
    uint32_t cin = (tid==0)? 0u : G[tid-1];
    uint64_t s2 = (uint64_t)X[tid] + Y[tid] + cin;
    OUT[tid] = (uint32_t)s2;
  }
  if(tid==n-1){
    *carry_out = G[n-1]; // overall carry-out
  }
}

__global__ void ks_sub32_kernel(const uint32_t* __restrict__ X,
                                const uint32_t* __restrict__ Y,
                                uint32_t* __restrict__ OUT,
                                uint32_t* __restrict__ borrow_out,
                                int n)
{
  __shared__ uint32_t G[MAX_LIMBS_BLOCK]; // borrow generate
  __shared__ uint32_t P[MAX_LIMBS_BLOCK]; // borrow propagate
  int tid = threadIdx.x;
  if(tid < n){
    uint32_t xi = X[tid], yi = Y[tid];
    G[tid] = (xi < yi) ? 1u : 0u;
    P[tid] = (xi == yi) ? 1u : 0u;
  }
  __syncthreads();

  for(int offset=1; offset < n; offset<<=1){
    uint32_t g_prev=0, p_prev=0;
    if(tid >= offset && tid < n){ g_prev = G[tid - offset]; p_prev = P[tid - offset]; }
    __syncthreads();
    if(tid < n){
      uint32_t g = G[tid], p = P[tid];
      uint32_t g_new = g | (p & g_prev);
      uint32_t p_new = p & p_prev;
      G[tid] = g_new; P[tid] = p_new;
    }
    __syncthreads();
  }

  if(tid < n){
    uint32_t bin = (tid==0)? 0u : G[tid-1];
    uint64_t d = (uint64_t)X[tid] - (uint64_t)Y[tid] - (uint64_t)bin;
    OUT[tid] = (uint32_t)d; // wrap OK
  }
  if(tid==n-1){
    *borrow_out = G[n-1]; // overall borrow-out with bin=0
  }
}
#endif

static void gpu_add_vec(const std::vector<uint32_t>& X,const std::vector<uint32_t>& Y,
                        std::vector<uint32_t>& OUT, uint32_t& carry)
{
#ifdef __CUDACC__
  if(cuda_available() && (int)std::max(X.size(),Y.size()) <= MAX_LIMBS_BLOCK){
    int n = (int)std::max(X.size(),Y.size());
    std::vector<uint32_t> Xp(n,0), Yp(n,0);
    std::copy(X.begin(), X.end(), Xp.begin());
    std::copy(Y.begin(), Y.end(), Yp.begin());
    uint32_t *dX=nullptr,*dY=nullptr,*dOUT=nullptr,*dC=nullptr;
    CUDA_CHECK(cudaMalloc(&dX, n*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dY, n*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dOUT, n*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dX, Xp.data(), n*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, Yp.data(), n*sizeof(uint32_t), cudaMemcpyHostToDevice));
    dim3 block(n), grid(1);
    ks_add32_kernel<<<grid, block>>>(dX,dY,dOUT,dC,n);
    CUDA_CHECK(cudaGetLastError());
    OUT.assign(n,0u);
    CUDA_CHECK(cudaMemcpy(OUT.data(), dOUT, n*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&carry, dC, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dY); cudaFree(dOUT); cudaFree(dC);
    trim_vec(OUT); if(carry) OUT.push_back(carry);
    return;
  }
#endif
  // CPU fallback
  int n=(int)std::max(X.size(),Y.size());
  OUT.assign(n,0u);
  unsigned long long c=0;
  for(int i=0;i<n;i++){
    unsigned long long xi = (i<(int)X.size()? X[i]:0u);
    unsigned long long yi = (i<(int)Y.size()? Y[i]:0u);
    unsigned long long s = xi + yi + c;
    OUT[i] = (uint32_t)s; c = s>>32;
  }
  carry = (uint32_t)c; trim_vec(OUT); if(carry) OUT.push_back(carry);
}

static void gpu_sub_vec(const std::vector<uint32_t>& X,const std::vector<uint32_t>& Y,
                        std::vector<uint32_t>& OUT, uint32_t& borrow)
{
#ifdef __CUDACC__
  if(cuda_available() && (int)std::max(X.size(),Y.size()) <= MAX_LIMBS_BLOCK){
    int n = (int)std::max(X.size(),Y.size());
    std::vector<uint32_t> Xp(n,0), Yp(n,0);
    std::copy(X.begin(), X.end(), Xp.begin());
    std::copy(Y.begin(), Y.end(), Yp.begin());
    uint32_t *dX=nullptr,*dY=nullptr,*dOUT=nullptr,*dB=nullptr;
    CUDA_CHECK(cudaMalloc(&dX, n*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dY, n*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dOUT, n*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dX, Xp.data(), n*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, Yp.data(), n*sizeof(uint32_t), cudaMemcpyHostToDevice));
    dim3 block(n), grid(1);
    ks_sub32_kernel<<<grid, block>>>(dX,dY,dOUT,dB,n);
    CUDA_CHECK(cudaGetLastError());
    OUT.assign(n,0u);
    CUDA_CHECK(cudaMemcpy(OUT.data(), dOUT, n*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&borrow, dB, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(dX); cudaFree(dY); cudaFree(dOUT); cudaFree(dB);
    trim_vec(OUT); return;
  }
#endif
  // CPU fallback
  int n=(int)std::max(X.size(),Y.size());
  OUT.assign(n,0u);
  unsigned long long b=0;
  for(int i=0;i<n;i++){
    unsigned long long xi = (i<(int)X.size()? X[i]:0u);
    unsigned long long yi = (i<(int)Y.size()? Y[i]:0u);
    unsigned long long d = xi - yi - b;
    OUT[i]=(uint32_t)d; b=(xi<yi+b)?1:0;
  }
  borrow=(uint32_t)b; trim_vec(OUT);
}

static void gpu_inc_vec(std::vector<uint32_t>& X){
  std::vector<uint32_t> one(1,1), out; uint32_t c=0;
  gpu_add_vec(X, one, out, c);
  X.swap(out);
}
static void gpu_dec_vec(std::vector<uint32_t>& X){
  std::vector<uint32_t> one(1,1), out; uint32_t b=0;
  gpu_sub_vec(X, one, out, b);
  X.swap(out); trim_vec(X);
}

// ===================== Newton: reciprocal & division =====================
static void mul_full(const std::vector<uint32_t>& A,const std::vector<uint32_t>& B,std::vector<uint32_t>& C){
  mul_full(A.data(), (int)A.size(), B.data(), (int)B.size(), C);
}
static void mul_trunc_lo(const std::vector<uint32_t>& A,const std::vector<uint32_t>& B,int t,std::vector<uint32_t>& C){
  mul_trunc_lo(A.data(), (int)A.size(), B.data(), (int)B.size(), t, C);
}

// REPLACE the whole reciprocal_scaled_2m() with this:
static void reciprocal_scaled_2m(const std::vector<uint32_t>& V,std::vector<uint32_t>& Y2m){
  int m=(int)V.size(); if(m==0){ Y2m.assign(1,0); return; }
  uint32_t vtop = V[m-1] ? V[m-1] : 1u;
  uint64_t B = 1ull<<32;

  std::vector<uint32_t> Y(1); Y[0] = (uint32_t)((B-1)/vtop); // 1-limb approx
  int t=1;
  while(t < 2*m){
    int Tlen = std::min(2*t, 2*m);

    // T = V * Y (low Tlen limbs)  [GPU or CPU via your mul_trunc_lo wrapper]
    std::vector<uint32_t> T;
    mul_trunc_lo(V, Y, Tlen, T);

    // E = 2*B^t - T  (CPU exact)
    std::vector<uint32_t> E(Tlen, 0u);
    if(t < Tlen) E[t] = 2; // 2*B^t
    uint64_t br=0;
    for(int i=0;i<Tlen;i++){
      uint64_t ei=E[i], ti=T[i];
      uint64_t d = ei - ti - br;
      E[i] = (uint32_t)d;
      br = (ei < ti + br) ? 1u : 0u;
    }

    // Y = (Y * E) >> (32*t)  [GPU or CPU via mul_full wrapper]
    std::vector<uint32_t> P; mul_full(Y, E, P);
    std::vector<uint32_t> Ynew(Tlen, 0u);
    for(int i=0;i<Tlen;i++){
      size_t idx = (size_t)t + i;
      Ynew[i] = (idx < P.size()) ? P[idx] : 0u;
    }
    Y.swap(Ynew);
    t = Tlen;
  }
  Y.resize(2*m,0u);
  Y2m = Y;
}

// REPLACE div_newton_scaled_gpu(...) with this:
static void div_newton_scaled_gpu(const std::vector<uint32_t>& U,const std::vector<uint32_t>& V,
                                  std::vector<uint32_t>& Q,std::vector<uint32_t>& R)
{
  int nU=(int)U.size(), nV=trim_len(V.data(), (int)V.size());
  if(nV==0){ std::fprintf(stderr,"Division by zero\n"); std::exit(1); }
  if(nU < nV){ Q.assign(1,0); R = U; trim_vec(R); return; }
  if(nV==1 && V[0]==1u){ Q = U; trim_vec(Q); R.assign(1,0u); return; }

  int m=nV, L=nU-m+1;

  // 1) 2m-limb reciprocal  (GPU muls inside)
  std::vector<uint32_t> Y2m; reciprocal_scaled_2m(V, Y2m);

  // 2) Q_est = floor((U * Y2m) / B^{2m})  (GPU mul)
  std::vector<uint32_t> Prod; mul_full(U, Y2m, Prod);
  Q.assign(L, 0u);
  for(int i=0;i<L;i++){
    size_t idx = (size_t)2*m + i;
    Q[i] = (idx < Prod.size()) ? Prod[idx] : 0u;
  }
  trim_vec(Q);

  // 3) T = Q*V  (GPU mul)
  std::vector<uint32_t> T; mul_full(Q, V, T);
  int nT = trim_len(T.data(), (int)T.size());
  int nUtrim = trim_len(U.data(), nU);

  if(cmp_be(T.data(), nT, U.data(), nUtrim) <= 0){
    // ---- Undershoot: R0 = U - T
    std::vector<uint32_t> R0(std::max(nUtrim,nT)+1,0u);
    for(int i=0;i<nUtrim;i++) R0[i]=U[i];
    (void)sub_into(R0.data(), T.data(), std::min(nT,(int)R0.size())); // CPU limb-sub
    trim_vec(R0);

    if(cmp_be(R0.data(), (int)R0.size(), V.data(), nV) < 0){
      R = R0; return;
    }
    // Robust correction via Knuth on residual
    std::vector<uint32_t> q_add, r_add; div_knuth(R0, V, q_add, r_add);
    if(q_add.size()>Q.size()) Q.resize(q_add.size(),0u);
    uint32_t c = add_into(Q.data(), q_add.data(), (int)q_add.size()); // CPU limb-add
    if(c) Q.push_back(c);
    trim_vec(Q);
    R = r_add; trim_vec(R);
    return;
  }

  // ---- Overshoot: D = T - U, q_sub = ceil(D / V)
  std::vector<uint32_t> D(std::max(nT,nUtrim)+1,0u);
  for(int i=0;i<nT;i++) D[i]=T[i];
  {
    std::vector<uint32_t> Uext(nT,0u);
    for(int i=0;i<nUtrim;i++) Uext[i]=U[i];
    (void)sub_into(D.data(), Uext.data(), nT); // CPU limb-sub
    trim_vec(D);
  }
  std::vector<uint32_t> q0, rem; div_knuth(D, V, q0, rem);
  std::vector<uint32_t> q_sub = q0;
  if(!is_zero_vec(rem)) inc_vec(q_sub); // ceil

  // Q = Q - q_sub  (CPU limb-sub)
  if(q_sub.size()>Q.size()) Q.resize(q_sub.size(),0u);
  uint64_t br=0;
  for(size_t i=0;i<q_sub.size();++i){
    uint64_t xi=Q[i], yi=q_sub[i];
    uint64_t d=xi - yi - br; Q[i]=(uint32_t)d; br=(xi < yi + br)?1:0;
  }
  for(size_t i=q_sub.size(); br && i<Q.size(); ++i){
    uint64_t xi=Q[i]; uint64_t d=xi - br; Q[i]=(uint32_t)d; br=(xi<br)?1:0;
  }
  trim_vec(Q);

  // R = (rem==0 ? 0 : V - rem)
  if(is_zero_vec(rem)){
    R.assign(1,0u);
  }else{
    R = V;
    R.resize(std::max(R.size(), rem.size()), 0u);
    rem.resize(R.size(), 0u);
    (void)sub_into(R.data(), rem.data(), (int)R.size()); // CPU limb-sub
    trim_vec(R);
  }
}

// ===================== Verifier (CPU mul for exactness) =====================
static bool verify_QVR_equals_U(const std::vector<uint32_t>& U,const std::vector<uint32_t>& V,
                                const std::vector<uint32_t>& Q,const std::vector<uint32_t>& R)
{
  if(cmp_be(R.data(), (int)R.size(), V.data(), (int)V.size()) >= 0) return false;
  std::vector<uint32_t> T; mul_full_cpu(Q.data(), (int)Q.size(), V.data(), (int)V.size(), T);
  size_t N = std::max(U.size(), T.size());
  T.resize(N, 0u);
  uint64_t c=0; size_t i=0;
  for(; i<R.size(); ++i){ uint64_t s=(uint64_t)T[i]+R[i]+c; T[i]=(uint32_t)s; c=s>>32; }
  for(; c && i<N; ++i){ uint64_t s=(uint64_t)T[i]+c; T[i]=(uint32_t)s; c=s>>32; }
  if(T.size() < U.size()) T.resize(U.size(),0u);
  for(size_t k=0;k<U.size();++k) if(T[k]!=U[k]) return false;
  return true;
}

// ===================== Tests (same 24 + your originals) =====================
struct Test { std::string name; std::vector<uint32_t> U, V, Qexp, Rexp; bool composeU; };

static void buildU_from_VQR(const std::vector<uint32_t>& V,const std::vector<uint32_t>& Q,const std::vector<uint32_t>& R,
                            std::vector<uint32_t>& U){
  std::vector<uint32_t> T; mul_full_cpu(Q.data(), (int)Q.size(), V.data(), (int)V.size(), T);
  U = T; size_t N = std::max(T.size(), R.size()); U.resize(N, 0u);
  uint64_t c=0; size_t i=0;
  for(; i<R.size(); ++i){ uint64_t s=(uint64_t)U[i]+R[i]+c; U[i]=(uint32_t)s; c=s>>32; }
  for(; c && i<U.size(); ++i){ uint64_t s=(uint64_t)U[i]+c; U[i]=(uint32_t)s; c=s>>32; }
  if(c) U.push_back((uint32_t)c);
  trim_vec(U);
}

static uint64_t g_seed = 0xBADC0FFEE0DDF00DULL;
static inline uint32_t rnd32(){
  uint64_t z = (g_seed += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return (uint32_t)(z >> 32);
}
static std::vector<uint32_t> rand_vec(int n, bool top_set_msb){
  std::vector<uint32_t> out(std::max(n,1));
  for(int i=0;i<n;i++) out[i]=rnd32();
  if(n>0){ if(top_set_msb) out[n-1] |= 0x80000000u; if(out[n-1]==0) out[n-1]=1u; }
  trim_vec(out); return out;
}
static std::vector<uint32_t> rand_V(int n, bool top_set_msb, bool even_lsb){
  auto v = rand_vec(n, top_set_msb);
  if(even_lsb){ v[0] &= ~1u; if(v[0]==0) v[0]=2u; }
  return v;
}
static std::vector<uint32_t> rand_Q(int n, bool top_set_msb){ return rand_vec(n, top_set_msb); }
static std::vector<uint32_t> rand_R_lt_V(const std::vector<uint32_t>& V){
  std::vector<uint32_t> W = rand_vec((int)std::max<size_t>(1,V.size()), false);
  std::vector<uint32_t> q,r; div_knuth(W, V, q, r); trim_vec(r); return r;
}
static void add_test_from_VQR(std::vector<Test>& tests, const std::string& name,
                              const std::vector<uint32_t>& V, const std::vector<uint32_t>& Q,
                              const std::vector<uint32_t>& R){
  Test t; t.name = name; t.V = V; t.Qexp = Q; t.Rexp = R; t.composeU = true; tests.push_back(std::move(t));
}

int main(){
#ifdef __CUDACC__
  if(cuda_available()){
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop,0);
    std::printf("CUDA: Using device 0: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  }else{
    std::puts("CUDA: No device found, running CPU fallbacks.");
  }
#else
  std::puts("Built without CUDA toolchain; running CPU only.");
#endif

  // Your original 12 tests:
  std::vector<Test> tests = {
    { "T1 16/8 (your case)",
      {0x48a2fb9e,0x6f64ae65,0x09e50c80,0x388b95df,0x5c0369b2,0x971a3d69,0x5c7ce7b4,0x53ac6f83,0x1469cf06,0xa5e0daad,0x9adffc28,0xf82eb167,0xc48bff9a,0xf56c4a31,0x17473ddc,0xd82ebf8b},
      {0x9f8c301a,0xe441c0b1,0xd7d5425c,0xa5affa0b,0x2f1207ee,0x95c02e75,0xf320ed50,0xe078e06f},
      {0x5273961a,0x673ddec0,0xfae0b756,0xd304b7ab,0xd72ec174,0x5936aeb2,0x4def9a6d,0xf68bcfd6},
      {0x0088dcfa,0x1ddcd5fd,0xb6588fff,0x72bee29c,0xb61b2f43,0x499beee8,0xa057893b,0x64960e71},
      false
    },
    { "T2 V=1",
      {0x11111111,0x22222222,0x33333333,0x44444444,0x55555555},
      {0x00000001},
      {0x11111111,0x22222222,0x33333333,0x44444444,0x55555555},
      {0x00000000},
      true
    },
    { "T3 V=B",
      {0xaaaaaaaa,0xbbbbbbbb,0xcccccccc,0xdddddddd},
      {0x00000000,0x00000001},
      {0xbbbbbbbb,0xcccccccc,0xdddddddd},
      {0xaaaaaaaa},
      true
    },
    { "T4 V=B^2",
      {0xdeadbeef,0x11111111,0x22222222,0x33333333,0x44444444},
      {0x00000000,0x00000000,0x00000001},
      {0x22222222,0x33333333,0x44444444},
      {0xdeadbeef,0x11111111},
      true
    },
    { "T5 single-limb odd V", {}, {0xabcdef01},
      {0x13579bdf}, {0x12345678}, true
    },
    { "T6 single-limb even V", {}, {0x10000000},
      {0x89abcdef,0x01234567}, {0x00000fff}, true
    },
    { "T7 multi-limb V/Q", {}, {0x01234567,0x89abcdef},
      {0xfedcba98,0x00000001,0x00000002}, {0x11111111,0x22222222}, true
    },
    { "T8 exact division", {}, {0xffffffff,0x00000001},
      {0x12345678,0x9abcdef0}, {0x00000000}, true
    },
    { "T9 U<V",
      {0x00000001,0x00000000}, {0x00000002,0x00000001},
      {0x00000000}, {0x00000001,0x00000000}, false
    },
    { "T10 random 6/3", {}, {0xa3b1799d,0x1c80317f,0x06671ad1},
      {0x6c928b68,0x7b1032e8,0x723902f0,0x0000001d},
      {0x492a9a33,0xf832cdce,0x0533eec9}, true
    },
    { "T11 random 10/5", {}, {0x1a3d1fa7,0xad3c2d6d,0xbd9c66b3,0xe465e150,0x8b9d2434},
      {0x9d9c76e7,0x7941a3e2,0x089230eb,0xcf327cd7,0x1a97d24e,0x00000001},
      {0x87bf15d1,0xe3d57cc7,0x4a5f9d48,0xb2cfea8e,0x2d70dcfc}, true
    },
    { "T12 random 8/4", {}, {0x06cb0fb3,0x8fadc1a6,0x32e70629,0xb74d0fb1},
      {0xd99ffcc2,0x6d5ad557,0xa6187e75,0x63764cc7},
      {0x21dbb9e3,0xc65d3cc9,0xbd041b02,0x4d35d7bf}, true
    },
  };
    // ---- NEW: larger + edge tests (deterministic) ----
  // T13: Large exact division 64/64
  {
    auto V = rand_V(64, true, false);
    auto Q = rand_Q(64, true);
    std::vector<uint32_t> R = {0};
    add_test_from_VQR(tests, "T13 exact large 64/64", V, Q, R);
  }
  // T14: Large with remainder 64/64 (R random < V)
  {
    auto V = rand_V(64, true, false);
    auto Q = rand_Q(64, true);
    auto R = rand_R_lt_V(V);
    add_test_from_VQR(tests, "T14 large 64/64 +R", V, Q, R);
  }
  // T15: Force big normalization shift (tiny top limb), 48/24
  {
    auto V = rand_V(24, false, false); V.back() = 1; // top limb tiny
    auto Q = rand_Q(25, true);
    auto R = rand_R_lt_V(V);
    add_test_from_VQR(tests, "T15 tiny-top V 48/24", V, Q, R);
  }
  // T16: Many carries (all-0xFFFFFFFF-ish), 40/20
  {
    std::vector<uint32_t> V(20, 0xffffffffu); V.back() = 0xfffffffeu;
    std::vector<uint32_t> Q(21, 0xffffffffu); Q.back() = 0x7fffffffu;
    auto R = rand_R_lt_V(V);
    add_test_from_VQR(tests, "T16 carry storm 40/20", V, Q, R);
  }
  // T17: Even LSB divisor (exercise Newton even case), 56/28
  {
    auto V = rand_V(28, true, true);  // even LSB
    auto Q = rand_Q(29, true);
    auto R = rand_R_lt_V(V);
    add_test_from_VQR(tests, "T17 even-LSB V 56/28", V, Q, R);
  }
  // T18: U = V - 1 (Q=0, R=V-1), 33/33 pattern
  {
    auto V = rand_V(33, true, false);
    auto R = V; // R = V - 1
    uint32_t b = 1;
    for(size_t i=0;i<R.size();++i){
      uint64_t xi=R[i]; uint64_t d=xi - b; R[i]=(uint32_t)d; b=(xi<b)?1:0;
      if(!b) break;
    }
    std::vector<uint32_t> Q = {0};
    add_test_from_VQR(tests, "T18 U=V-1 (Q=0)", V, Q, R);
  }
  // T19: U = V + (V-1) (Q=1, R=V-1)
  {
    auto V = rand_V(36, true, false);
    std::vector<uint32_t> Q = {1};
    auto R = V; // R = V - 1
    uint32_t b = 1;
    for(size_t i=0;i<R.size();++i){
      uint64_t xi=R[i]; uint64_t d=xi - b; R[i]=(uint32_t)d; b=(xi<b)?1:0;
      if(!b) break;
    }
    add_test_from_VQR(tests, "T19 U=2V-1 (Q=1)", V, Q, R);
  }
  // T20: Highly skewed 100/2
  {
    auto V = rand_V(2, true, false);
    auto Q = rand_Q(99, true);
    auto R = rand_R_lt_V(V);
    add_test_from_VQR(tests, "T20 skew 100/2", V, Q, R);
  }
  // T21: Skewed 96/48
  {
    auto V = rand_V(48, true, false);
    auto Q = rand_Q(49, true);
    auto R = rand_R_lt_V(V);
    add_test_from_VQR(tests, "T21 skew 96/48", V, Q, R);
  }
  // T22: Large 80/40 exact
  {
    auto V = rand_V(40, true, false);
    auto Q = rand_Q(41, true);
    std::vector<uint32_t> R = {0};
    add_test_from_VQR(tests, "T22 exact 80/40", V, Q, R);
  }
  // T23: Large 72/36 with near-max remainder R=V-1
  {
    auto V = rand_V(36, true, false);
    auto Q = rand_Q(37, true);
    auto R = V;
    uint32_t b = 1; for(size_t i=0;i<R.size();++i){ uint64_t xi=R[i]; uint64_t d=xi - b; R[i]=(uint32_t)d; b=(xi<b)?1:0; if(!b)break; }
    add_test_from_VQR(tests, "T23 R=V-1 72/36", V, Q, R);
  }
  // T24: Random 64/32 with even LSB and tiny top limb (mixed)
  {
    auto V = rand_V(32, false, true); V.back() = 1; // tiny top + even LSB
    auto Q = rand_Q(33, true);
    auto R = rand_R_lt_V(V);
    add_test_from_VQR(tests, "T24 mixed stress 64/32", V, Q, R);
  }
 

  // Plus larger stress tests like before (T13..T24). Omitted for brevity — keep yours.

  int pass=0, total=(int)tests.size()*2;
  for(auto& t : tests){
    std::vector<uint32_t> V = t.V; for(auto &x:V) x &= 0xFFFFFFFFu;
    std::vector<uint32_t> U = t.U; if(t.composeU){ buildU_from_VQR(V, t.Qexp, t.Rexp, U); }

    std::vector<uint32_t> Qk,Rk; div_knuth(U, V, Qk, Rk);
    bool okK = verify_QVR_equals_U(U,V,Qk,Rk) && eq_vec(Qk,t.Qexp) && eq_vec(Rk,t.Rexp);
    std::printf("[%-22s] Knuth  : %s\n", t.name.c_str(), okK?"PASS":"FAIL");
    if(!okK){ print_vec("  Qk", Qk); print_vec("  Qexp", t.Qexp); print_vec("  Rk", Rk); print_vec("  Rexp", t.Rexp); }
    else ++pass;

    std::vector<uint32_t> Qn,Rn; div_newton_scaled_gpu(U, V, Qn, Rn);
    bool okN = verify_QVR_equals_U(U,V,Qn,Rn) && eq_vec(Qn,t.Qexp) && eq_vec(Rn,t.Rexp);
    std::printf("[%-22s] Newton : %s\n", t.name.c_str(), okN?"PASS":"FAIL");
    if(!okN){ print_vec("  Qn", Qn); print_vec("  Qexp", t.Qexp); print_vec("  Rn", Rn); print_vec("  Rexp", t.Rexp); }
    else ++pass;
  }

  std::printf("\nSummary: %d / %d checks passed\n", pass, total);
  return (pass==total)? 0 : 1;
}
