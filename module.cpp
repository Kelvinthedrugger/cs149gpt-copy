#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
//#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
        const int &sizeX, const int &sizeY, const int &sizeZ) {
  return tensor[((x * sizeX + y) * sizeY + z) * sizeZ + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
  tensor[((x * sizeX + y) * sizeY + z) * sizeZ + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We
 * have also created O and QK^t Tensors that are formatted as vectors. After you
 * have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it
 * this way - if I asked my dnn a question and it output 5 different answers it
 * had a batch size of 5. These samples are independent of each other and thus
 * can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This
 * effectively allows each head to operate the same attention algorithm, but
 * each with each head using different hyperparameters. These allow each head to
 * have their own definition of what relevance is when looking at a token. These
 * heads can operate independently of one another and thus can be parallelized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the
 * number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per
 * attention head. Let's say I encoded a word using the follow (length, number
 * of vowels, has a capital letters). The embedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D
       accessors

         for (int i = 0; i < N; i++) {
             for (int j = 0; j < N; j++) {
               float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
               twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */

    // -------- YOUR CODE HERE  -------- //
    // loop over Batch Size
    for (int b = 0; b < B; b++) {
      // loop over Heads
      for (int h = 0; h < H; h++) {
        // loop over Sequence Length
        // Q * K_t
        // row
        for (int row = 0; row < N; row++) {
          for (int col = 0; col < N; col++) {
            float qk_t = 0.0f;
            // loop over Embedding Dimensionality
            for (int mid = 0; mid < d; mid++) {
              float q = fourDimRead(Q, b, h, row, mid, H, N, d);
              float k_t = fourDimRead(K, b, h, col, mid, H, N, d);
              // accumulate Q dot K_t for each 1 iteration
              qk_t += q * k_t;
            }
            // write Q dot K_t to O
            twoDimWrite(QK_t, row, col, N, qk_t);
            //printf("%d, %d, %.9f\n", row, col, qk_t);
            //if (col < d)
            //  fourDimWrite(O, b, h, row, col, N, d, H, qk_t);
          }
        }
        // softmax()
        for (int row = 0; row < N; row++) {
          float rowsum = 0.0f;
          for (int col = 0; col < N; col++) {
            float ele = twoDimRead(QK_t, row, col, N);
            ele = exp(ele); // cpp intrinsic exp()?
            rowsum += ele;
            twoDimWrite(QK_t, row, col, N, ele); // write back so that it's exponential'ed
            //printf("%d, %d, %.9f, %.9f\n", row, col, ele, rowsum);
          }
          // writeback the normalized elements to QK_t
          for (int col = 0; col < N; col++) {
            float ele = twoDimRead(QK_t, row, col, N) / rowsum;
            twoDimWrite(QK_t, row, col, N, ele);
            //printf("%d, %d, %.9f\n", row, col, ele);
          }
        }
        // dot V
        // row
        for (int row = 0; row < N; row++) {
          // loop over Embedding Dimensionality
          for (int col = 0; col < d; col++) {
            float val = 0.0f;
            for (int mid = 0; mid < N; mid++) {
              float ele = twoDimRead(QK_t, row, mid, N);
              float v = fourDimRead(V, b, h, mid, col, H, N, d);
              val += ele * v;
            }
            //printf("%d, %d, %.9f\n", row, col, val);
            fourDimWrite(O, b, h, row, col, H, N, d, val);
          }
        }
      }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

int min(int a, int b) { return std::min(a, b); }

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    // copied from part1: edit only a part of it
    //  , so that we'll know if things goes off
    // loop over Batch Size
    for (int b = 0; b < B; b++) {
      // loop over Heads
      for (int h = 0; h < H; h++) {
        // loop over Sequence Length
        // Q * K_t
        // row
        // done, but it's slower than QK from part 1
        //  i'm guessing it's because Q & K are accessed
        //  in row major originally, so this blocked version
        //  doesn't help a lot. consider to adjust the direction of access
        // from part 1 (190 ms in total)
        for (int row = 0; row < N; row++) {
          for (int col = 0; col < N; col++) {
            float qk_t = 0.0f;
            // loop over Embedding Dimensionality
            // twoDimWrite(QK_t, row, col, N, qk_t); // init 1st element as 0
            for (int mid = 0; mid < d; mid++) {
              float q = fourDimRead(Q, b, h, row, mid, H, N, d);
              float k_t = fourDimRead(K, b, h, col, mid, H, N, d);
              //   accumulate Q dot K_t for each 1 iteration
              qk_t += q * k_t;
            }
            // write Q dot K_t to O
            // printf("%d, %d, %.9f\n", row, col, qk_t);
            twoDimWrite(QK_t, row, col, N, qk_t);
          }
        }
        int tileSize = 4;
        // do i have to init QK_t with 0?
        // for (int row = 0; row < N; row += tileSize) {
        // for (int col = 0; col < N; col+=tileSize) {
        //  float qk_t = 0.0f;
        // loop over Embedding Dimensionality
        // part 2: 2-d tiling (260 ms)
        /*for (int mid = 0; mid < d; mid+=tileSize) {
          //   accumulate Q dot K_t for each 1 iteration
          //   iterate inside the tile
          for (int rowidx = row; rowidx < min(N, row + tileSize);
               rowidx++) {
            for (int colidx = col; colidx < min(N, col + tileSize);
                 colidx++) {
              // write 0 if mid == 0 ? don't have to, since
              // the memory has been statically allocated
              // -> init'ed as 0 by default
              // read value from (possibly written previously) tile
              qk_t = twoDimRead(QK_t, rowidx, colidx, N);
              for (int mididx = mid; mididx < min(d, mid + tileSize);
                   mididx++) {
                float q = fourDimRead(Q, b, h, rowidx, mididx, H, N, d);
                float k_t = fourDimRead(K, b, h, colidx, mididx, H, N, d);
                qk_t += q * k_t;
              }
              // write to corresponding tile
              twoDimWrite(QK_t, rowidx, colidx, N, qk_t);
            }
          }
        }*/
        // TODO implement 1-d tiling, see if the advantage of row-major plays
        // out
        /*for (int rowidx = row; rowidx < min(N, row + tileSize); rowidx++)
        { for (int colidx = col; colidx < min(N, col + tileSize); colidx++)
        { qk_t = 0.0f; for (int mid = 0; mid < N; mid++) { float q =
        fourDimRead(Q, b, h, rowidx, mid, H, N, d); float k_t =
        fourDimRead(K, b, h, colidx, mid, H, N, d); qk_t += q * k_t;
            }
            twoDimWrite(QK_t, rowidx, colidx, N, qk_t);
          }
        }*/

        // write Q dot K_t to O
        // }
        //}
        // softmax()
        for (int row = 0; row < N; row++) {
          float rowsum = 0.0f;
          for (int col = 0; col < N; col++) {
            float ele = twoDimRead(QK_t, row, col, N);
            ele = exp(ele); // cpp intrinsic exp()?
            rowsum += ele;
            twoDimWrite(QK_t, row, col, N, ele); // write back so that it's exponential'ed
            //printf("%d, %d, %.9f, %.9f\n", row, col, ele, rowsum);
          }
          // writeback the normalized elements to QK_t
          for (int col = 0; col < N; col++) {
            float ele = twoDimRead(QK_t, row, col, N) / rowsum;
            twoDimWrite(QK_t, row, col, N, ele);
            //printf("%d, %d, %.9f\n", row, col, ele);
          }
        }
        // dot V
        // row
        // done
        // int tileSize = 4; // moved to above
        for (int row = 0; row < N; row += tileSize) {
          // loop over Embedding Dimensionality
          float val = 0.0f;
          for (int col = 0; col < d; col += tileSize) {
            for (int mid = 0; mid < N; mid += tileSize) {
              // float ele = twoDimRead(QK_t, row, mid, N);
              // float v = fourDimRead(V, b, h, mid, col, H, N, d);
              // val += ele * v;
              // iterate inside the tile
              for (int rowidx = row; rowidx < min(N, row + tileSize);
                   rowidx++) {
                for (int colidx = col; colidx < min(d, col + tileSize);
                     colidx++) {
                  // read value from (possibly written previously) tile
                  val = fourDimRead(O, b, h, rowidx, colidx, H, N, d);
                  for (int mididx = mid; mididx < min(N, mid + tileSize);
                       mididx++) {
                    float ele = twoDimRead(QK_t, rowidx, mididx, N);
                    float v = fourDimRead(V, b, h, mididx, colidx, H, N, d);
                    val += ele * v;
                  }
                  // write to corresponding tile
                  fourDimWrite(O, b, h, rowidx, colidx, H, N, d, val);
                }
              }
            }
            // printf("%d, %d, %.9f\n", row, col, val);
            //  fourDimWrite(O, b, h, row, col, H, N, d, val);
          }
        }
      }
    }



    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    for (int b = 0; b < B; b++){
        //loop over heads
        for (int h = 0; h < H; h++){
          // compute Q dot K_t & exp'ed it & accumulate rowsum
          for (int row = 0; row < N; row++) {
            // YRow is moved inside so each OpenMP thread gets a local copy.
            // at::Tensor ORowTensor =
            // temp.index({torch::indexing::Slice(omp_get_thread_num(),
            // torch::indexing::None)}); std::vector<float> ORow =
            // formatTensor(ORowTensor); YOUR CODE HERE
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
        }
    }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor,
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    // copy from part 3
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);
    // 1D of size N
    std::vector<float> ORow = formatTensor(ORowTensor);
    for (int b = 0; b < B; b++) {
      // loop over heads
      for (int h = 0; h < H; h++) {
        /*for (int row = 0; row < N; row += Br) {
          for (int r = 0; r < Br; r++) {
            int rowaddr = row + r;
            float qi, kj, vj, qk_t = 0.0f;
            for (int col = 0; col < N; col += Bc) {
              for (int c = 0; c < Bc; c++) {
                int coladdr = col + c;
                for (int mid = 0; mid < d; mid++) {
                  if (rowaddr < N) {
                    qi = fourDimRead(Qi, b, h, rowaddr, mid, H, N, d);
                    li[r] = l[rowaddr];
                  }
                  if (coladdr < N) {
                    kj = fourDimRead(Kj, b, h, coladdr, mid, H, N, d);
                    vj = fourDimRead(Vj, b, h, coladdr, mid, H, N, d);
                  }
                  qk_t += qi * kj;
                }
              }
            }
          }
        }*/
        // compute Q dot K_t & exp'ed it & accumulate rowsum
        for (int row = 0; row < N; row++) {
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
      }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
