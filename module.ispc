
// copied from part 3
// can pytorch tensor be used here?

extern void one(
  std::vector<float> &Q, std::vector<float> &K, 
  std::vector<float> &V, std::vector<float> &O, /*O: rth row of ORow*/
  int b, int h, int H, int N, int d) {
  //at::Tensor ORowTensor = 
    temp.index({torch::indexing::Slice(omp_get_thread_num(),torch::indexing::None)}); 
  //std::vector<float> ORow = formatTensor(ORowTensor); 
  // above 2 lines : temp: size of (num_of_threads, N)
  //  -> easy for this function to call each row of it & run in parallel
  // YOUR CODE HERE
  // Q dot K_t
  float rowsum = 0.0f; // for softmax
  for (int col = 0; col < N; col++) {
    float qk_t = 0.0f;
    for (int mid = 0; mid < d; mid++) {
      float q = fourDimRead(Q, b, h, row, mid, H, N, d);
      float k_t = fourDimRead(K, b, h, col, mid, H, N, d);
      qk_t += q * k_t;
    }
    float exp_qkt = exp(qk_t);
    rowsum += exp_qkt;
    ORow[col] = exp_qkt;
  }
  // compute softmax
  for (int col = 0; col < N; col++) {
    ORow[col] /= rowsum;
  }
  // dot V
  for (int col = 0; col < d; col++) {
    float val = 0.0f;
    for (int mid = 0; mid < N; mid++) {
      // load p,v -> dot -> write back
      float p = ORow[mid];
      float v = fourDimRead(V, b, h, mid, col, H, N, d);
      val += p * v;
    }
    fourDimWrite(O, b, h, row, col, H, N, d, val);
  }
}
