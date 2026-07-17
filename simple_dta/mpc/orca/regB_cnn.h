// regB_cnn.h -- regB (dilated CNN+CNN, Davis 0.848) as a sytorch model, to add to
// EzPC/GPU-MPC/experiments/orca/cnn.h so Orca can BENCHMARK its GPU-FSS online time.
//
// Orca runs models by name from cnn.h on ZEROED input (timing is input-independent),
// so this measures regB's secure-inference latency -- compare to the 2.6s MP-SPDZ
// CPU online number (see ../ONLINE_TIME.md).
//
// Mapping from the PyTorch CNNDTA (models.py) to sytorch:
//   * embedding -> a 1x1 Conv2d over one-hot channels (V=vocab), OR feed the pre-
//     embedded [B, embed_dim, 1, L] tensor (Orca uses zeroed input either way).
//   * Conv1d(in,out,k, dilation=d) -> Conv2d(in,out, {1,k}, stride 1, dilation {1,d}).
//   * global max-pool over the length axis -> MaxPool2d over {1, L'} (Orca supports it).
//   * two towers (drug + protein) run, their pooled vectors concat, then the FC head.
//
// TEMPLATE: sytorch's exact class names / ctor signatures move between versions.
// Reconcile against EzPC/GPU-MPC/ext/sytorch/include and the existing getCNN cases
// (VGG/ResNet/AlexNet) in cnn.h. regB dims are exact.
//
// regB (Davis): embed_dim=128, drug L=85, protein L=1200 (one-hot vocab: drug 65,
// protein 26). Conv output lengths (no padding): drug 85->82->79->76 (k4),
// protein 1200->1193->1179->1151 (k8, dilations 1,2,4). Head 192->1024->512->1.

#pragma once
#include <sytorch/module.h>   // adjust to your sytorch include path

template <typename T>
class RegB : public SytorchModule<T> {
    using SytorchModule<T>::add;
    using SytorchModule<T>::concat;

    // --- drug tower (SMILES CNN): 16 -> 32 -> 48, kernel 4, no dilation ---
    Conv2D<T> *d_embed;                       // 1x1 conv: one-hot(65) -> 128
    Conv2D<T> *d1, *d2, *d3;                  // (128->16)->(16->32)->(32->48), k={1,4}
    ReLU<T> *dr1, *dr2, *dr3;
    MaxPool2D<T> *d_pool;                     // over {1, 76}

    // --- protein tower (CNN): 32 -> 64 -> 96, kernel 8, dilations 1,2,4 ---
    Conv2D<T> *p_embed;                       // 1x1 conv: one-hot(26) -> 128
    Conv2D<T> *p1, *p2, *p3;                  // dilation {1,1},{1,2},{1,4}
    ReLU<T> *pr1, *pr2, *pr3;
    MaxPool2D<T> *p_pool;                     // over {1, 1151}

    // --- head: concat(48+... wait 48 drug + 96 protein = 144)? regB head_in = 48+96 ---
    // NOTE regB head first layer is 192-wide in the PyTorch summary because drug tower
    // out=96 too in the *default* build; regB uses drug 16,32,48 -> 48 and protein
    // 32,64,96 -> 96, so concat = 144. Confirm head_in from your checkpoint's first
    // FC weight shape and set FC1 accordingly (144 or 192).
    FC<T> *fc1, *fc2, *fc3;                   // 144 -> 1024 -> 512 -> 1
    ReLU<T> *hr1, *hr2;

public:
    RegB() {
        d_embed = new Conv2D<T>(65, 128, {1, 1}, 0);
        d1 = new Conv2D<T>(128, 16, {1, 4}, 0);  dr1 = new ReLU<T>();
        d2 = new Conv2D<T>(16, 32, {1, 4}, 0);   dr2 = new ReLU<T>();
        d3 = new Conv2D<T>(32, 48, {1, 4}, 0);   dr3 = new ReLU<T>();
        d_pool = new MaxPool2D<T>({1, 76}, 0, {1, 1});

        p_embed = new Conv2D<T>(26, 128, {1, 1}, 0);
        // Conv2D dilation arg position is version-specific; set dilation {1,1},{1,2},{1,4}:
        p1 = new Conv2D<T>(128, 32, {1, 8}, 0);  pr1 = new ReLU<T>();   // dilation {1,1}
        p2 = new Conv2D<T>(32, 64, {1, 8}, 0);   pr2 = new ReLU<T>();   // dilation {1,2}
        p3 = new Conv2D<T>(64, 96, {1, 8}, 0);   pr3 = new ReLU<T>();   // dilation {1,4}
        p_pool = new MaxPool2D<T>({1, 1151}, 0, {1, 1});

        fc1 = new FC<T>(144, 1024);  hr1 = new ReLU<T>();
        fc2 = new FC<T>(1024, 512);  hr2 = new ReLU<T>();
        fc3 = new FC<T>(512, 1);
    }

    Tensor<T> &_forward(Tensor<T> &drug, Tensor<T> &prot) {
        auto &d = dr3->forward(d3->forward(dr2->forward(d2->forward(
                  dr1->forward(d1->forward(d_embed->forward(drug)))))));
        auto &dv = d_pool->forward(d);                       // [B,48,1,1]
        auto &p = pr3->forward(p3->forward(pr2->forward(p2->forward(
                  pr1->forward(p1->forward(p_embed->forward(prot)))))));
        auto &pv = p_pool->forward(p);                       // [B,96,1,1]
        auto &x = concat(dv, pv);                            // [B,144]
        return fc3->forward(hr2->forward(fc2->forward(hr1->forward(fc1->forward(x)))));
    }
};

// In cnn.h getCNN<T>(modelName): add
//   else if (name == "RegB") return new RegB<T>();
// and register "RegB" in the experiment runner's model list, bitwidth 64, scale 13.
