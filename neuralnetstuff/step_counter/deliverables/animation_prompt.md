# Prompt: animate how this step counter works

Copy everything below the line into a new Claude chat.

---

I built a step counter: an IMU (motion sensor) strapped to my ankle, and a small PyTorch neural network that counts my steps and reports how confident it is. Please make an **animated explainer** of how the whole system works, as a single self-contained HTML file using JavaScript with Canvas or SVG and no external libraries. Add play/pause and next/previous scene buttons, and a progress bar showing the current scene. Aim for about 4–6 minutes if played straight through. Each scene needs a short on-screen caption (1–2 sentences) explaining what is happening.

**Accuracy matters more than flash.** Every number, layer size and formula below is the real one from my project. Use them exactly, and don't invent extra layers, values or results. Where you need example signal values, stay within the ranges I give.

**Visual style:** dark "instrument panel" background (#121211, panels #1a1a19), light text (#f0efec, secondary #c3c2b7). Use these colors consistently for the three body axes and for total rotation:
- S axis (up the shin) = blue #3987e5
- F axis (forward) = orange #d95926
- L axis (sideways) = green #199e70
- |ω| (total rotation speed) = amber #c98500

For any magnitude heatmap, use one hue (blue, rgba(57,135,229, alpha)) from dark to bright. Never use a rainbow.

## Scene 1: Hardware and data flow (~25 s)

- An **MPU-6050** IMU (3-axis accelerometer + 3-axis gyroscope) and an **Arduino UNO Q** board, both strapped to the **outside of the right ankle**. The battery is on the thigh.
- The UNO Q has two processors: an **STM32U585 microcontroller** and a **Qualcomm QRB2210 Linux processor**.
- Data path, animated as packets moving left to right:
  1. The MPU-6050 is read over **I2C at 400 kHz** by the STM32, **100 times per second** (the MPU samples internally at 1 kHz with a ~44 Hz low-pass filter).
  2. The STM32 groups **10 samples** at a time and sends them over the **Bridge** (MessagePack remote procedure calls over a 115200-baud serial link) to Python on the Linux side.
  3. Python publishes over **Wi-Fi via MQTT**, grouped into **4 messages per second of about 25 samples each**. (The public broker drops messages above ~8 per second.)
  4. A laptop receives the stream and runs everything else: recording, labeling, training, inference and the UI.
- Each sample is 8 numbers: `seq, t_ms, ax, ay, az, gx, gy, gz`. The 6 sensor values are raw 16-bit counts (accel: 2048 counts per g at ±16 g; gyro: 16.4 counts per °/s at ±2000 °/s).

## Scene 2: What the signals look like (~30 s)

- Convert to physical units, then rotate into the **body frame**:
  - **S** = up the shin (−x of the sensor)
  - **F** = forward (−y)
  - **L** = toward the body's left, i.e. the inner side of the right ankle (+z)

  S, F, L form a right-handed frame.
- **Standing still:** the accelerometer reads **S ≈ +1 g** (it measures the upward push that holds it against gravity), F ≈ L ≈ 0. The gyro reads ~0 after subtracting its bias (about −7.5 °/s on the x axis, measured during a 5 s standing calibration).
- **Walking:** animate a scrolling 10 s plot. Once per **right-foot stride** (normal walking: one stride every **~1.13 s**; a stride = two steps), the **L gyro dips sharply to about −300 to −450 °/s**. That dip is the **mid-swing**, when the shin swings forward fastest. About 0.1–0.4 s later the foot lands and the **acceleration magnitude spikes to 2–6 g** (heel strike).
- Show three contrasting cases:
  - **Seated leg swing:** smooth L oscillations of about ±150 °/s but **no impact** (|a| stays ≈ 1.3 g). It's not a step, and a naive peak detector gets fooled.
  - **Turning on the spot:** each turning step is a **big spike on the S axis** (the shin twisting, up to ~±390 °/s) with almost no L dip.
  - **Side and backward steps:** bursts on S and F, and *upward* L peaks.

## Scene 3: Labels and windows (~30 s)

- **Ground truth:** during each block (walk 40 steps, potter for 30 s, and so on) I counted my steps (both feet). Then, in a labeling tool, I marked **one event per right-foot stride**:
  - forward steps at the **minimum of the L gyro**;
  - turning, side and backward steps at the **peak of |ω|**, the total rotation speed √(S² + F² + L²) of the gyro.

  A block can only be saved if **|2 × strides − counted steps| ≤ 1** (the ±1 because a bout can start and end on the same foot).
- **The network's input:** a window of **256 samples = 2.56 s** × **6 channels** (accel S, F, L in g; gyro S, F, L in °/s). Each channel is normalized with mean/std from the training data:
  - mean = [1.035, −0.072, 0.146, 3.63, −1.56, −0.83]
  - std = [0.35, 0.47, 0.23, 50.2, 37.3, 87.1]
- **The target** (same length, 256): **0 everywhere, with a Gaussian bump (σ = 5 samples = 50 ms, peak 1.0) centred on each labeled stride**.
- **A mask** marks which samples have trustworthy labels (inside counted blocks). The loss ignores everything else.
- **Training windows** are random crops around labeled samples, **3,000 per epoch**, with augmentation: a random rotation of up to **±10°** (applied identically to the accel and gyro vectors, like slightly re-strapping the sensor) and amplitude scaling of **±10%**.

## Scene 4: The network architecture (~40 s)

A **dilated 1D convolutional network** (PyTorch `StepCNN`) with **25,937 trainable parameters**. Every layer keeps the time length 256 ("same" padding), so every layer's output is itself a signal over time. Draw it as a stack of horizontal bands (channels × time), with each layer's tensor shape labeled:

| layer | operation | in → out channels | kernel | dilation | padding | parameters | output shape |
|---|---|---|---|---|---|---|---|
| input | | 6 | | | | | (batch, 6, 256) |
| conv1 | Conv1d → ReLU → Dropout(0.1) | 6 → 16 | 7 | 1 | 3 | 688 | (batch, 16, 256) |
| conv2 | Conv1d → ReLU → Dropout(0.1) | 16 → 32 | 7 | 2 | 6 | 3,616 | (batch, 32, 256) |
| conv3 | Conv1d → ReLU → Dropout(0.1) | 32 → 32 | 7 | 4 | 12 | 7,200 | (batch, 32, 256) |
| conv4 | Conv1d → ReLU → Dropout(0.1) | 32 → 32 | 7 | 8 | 24 | 7,200 | (batch, 32, 256) |
| conv5 | Conv1d → ReLU → Dropout(0.1) | 32 → 32 | 7 | 16 | 48 | 7,200 | (batch, 32, 256) |
| head | Conv1d (1×1), no activation | 32 → 1 | 1 | 1 | 0 | 33 | (batch, 1, 256) → squeeze → logits (batch, 256) |

- **Animate dilation:** a 7-tap kernel whose taps are 1, 2, 4, 8, then 16 samples apart in successive layers.
- **Animate the receptive field growing:** one output sample depends on 7, 19, 43, 91, then **187 input samples** (1 + 6 × (1 + 2 + 4 + 8 + 16) = 187, i.e. **1.87 s**). That's longer than one stride even at slow walking (~1.4 s), so the network always sees at least a full stride around each moment.
- Show the weight count for one layer, e.g. conv3: 32 × 32 × 7 weights + 32 biases = 7,200.

## Scene 5: The forward pass (~30 s)

- Take one 256-sample window of walking. Show it flowing through the layers: each layer's 16 or 32 channels light up as a heatmap over time.
  - Early layers respond to local wiggles.
  - Deeper layers respond to whole stride-shaped patterns.
- The head outputs **256 logits**. A **sigmoid** turns them into probabilities: **p(stride) at every sample**. The output curve has sharp peaks (~0.97) at each mid-swing and is ~0 elsewhere.
- Show the PyTorch view briefly: `x` is a **tensor** of shape `(64, 6, 256)` for a batch of 64 windows, and `logits = model(x)` has shape `(64, 256)`.

## Scene 6: The loss (~30 s)

**Weighted binary cross-entropy** between the logits and the Gaussian target, at every sample:

```
loss_t = −[ y_t · log σ(z_t) + (1 − y_t) · log(1 − σ(z_t)) ]
weight_t = (1 + 20 · y_t) · mask_t
loss = Σ_t weight_t · loss_t  /  Σ_t mask_t
```

- **Why the weight of 20:** strides are rare (about 1 sample in 100 is near a stride). Without it, the network could score well by predicting "no stride" everywhere.
- **Why the mask:** samples outside labeled blocks might contain real but uncounted strides, so they must not be taught as "no stride".
- Animate: predicted curve vs target curve. The gap is shaded and scaled by the weight, and summed into one number.

## Scene 7: Backpropagation and the optimizer (~45 s)

- Show PyTorch's **autograd** idea:
  - During the forward pass, every operation is recorded in a **computation graph**: conv → ReLU → dropout → … → sigmoid/BCE.
  - `loss.backward()` walks that graph **in reverse**, applying the **chain rule**. Each layer receives ∂loss/∂(its output), multiplies by its local derivative, and passes the result on.
  - Animate gradients as particles flowing **backwards** from the loss, through the head, then conv5 … conv1:
    - At each ReLU, gradient passes only where the input was > 0.
    - At each dropout, it passes only through the units that were kept.
    - Every weight ends up with its own gradient `.grad`, the same shape as the weight.
- **The Adam optimizer** (`torch.optim.Adam`, learning rate **0.001**, weight decay **0.0001**, β₁ = 0.9, β₂ = 0.999): for each weight, a running average of its gradient (m) and of its squared gradient (v); the step is w ← w − lr · m̂ / (√v̂ + ε). Show weights nudging slightly after each `optimizer.step()`, after `optimizer.zero_grad()` clears the old gradients.
- **The training loop:** **30 epochs × 47 batches of 64 windows = 1,410 optimizer steps**. Show the loss falling from about **1.0 to about 0.5** (it never reaches 0, because the soft Gaussian targets can't be matched exactly). Each model trains in about **17 s on an Apple GPU** (PyTorch's "mps" device).

## Scene 8: From probabilities to a step count (~30 s)

- On a whole recording, slide the 256-sample window along with a hop of **64 samples** offline (**50** in the live app). Each window's 256 outputs are **weighted by a Hann window** (edges count less, because they see less context) and averaged where windows overlap. The result is one continuous probability curve.
- **Peak picking:** every local maximum above **0.5**, at least **0.3 s** apart, is a stride.
- **Steps = 2 × strides.**
- In the live app, a stride only becomes final once it is **128 samples (1.28 s)** behind the newest data, because the network needs to see what comes after it. Show that "not final yet" zone at the right edge.

## Scene 9: Uncertainty, turning a count into a count ± interval (~50 s)

- **Ensemble:** **5 copies** of the network, trained identically except for their random seeds (initial weights, window order, augmentation). Average their probability curves; where the copies disagree, show a band.
- For a stretch of time, find the candidate peaks k (height ≥ 0.05) on the mean curve, each with probability p_k. Three sources of variance:
  1. **Poisson-binomial:** each candidate is an independent yes/no event, so Var_PB = Σ p_k (1 − p_k). A peak at p = 0.5 adds the most (0.25); confident peaks (p ≈ 0 or 1) add almost nothing.
  2. **Ensemble disagreement:** Var_ens = the variance of the 5 networks' own stride counts.
  3. **Doubling:** steps = 2 × strides is off by up to 1 per bout (it depends on which foot starts and which finishes), so Var_dbl = bouts × σ_δ², with **σ_δ² = 0.054** measured from my counts.
- **Combine** (×4 converts strides to steps):

```
σ = √( 4 · (Var_PB + Var_ens) + Var_dbl ),  never below 0.5 steps
interval = N̂ ± q · σ
```

- **Calibrating q (conformal prediction):** on held-out data, measure how far off the counts really were, in units of σ, and take the 90th percentile. Here about 90% of blocks were counted *exactly* right, so that percentile came out as 0, which would mean a zero-width interval. That still covers 90% of blocks *on average*, but only 60% of the hard ones. So the rule is **q = max(conformal q, Gaussian z)**: calibration may widen the model's interval but never shrink it. That gives **q₉₀ = 1.645** (and q₅₀ = 0.674, q₈₀ = 1.282, q₉₅ = 1.960).
- Animate, for one 40-step walk, the count 40 with its interval **40 ± 2.1** forming around it.

## Scene 10: Results and the final count (~30 s)

Tested on two sessions the network never saw (S009, scripted; S010, everyday movement), against a hand-tuned rule-based detector (B2: low-passed L-gyro peak detection plus a heel-strike impact check):

| | counted steps | neural network ensemble | rule-based B2 |
|---|---|---|---|
| S009 (walking, running, stairs, shuffle) | 268 | **268** (exact) | 272 |
| S010 (pottering, turning on the spot, sit-to-stand…) | 90 | **88** (−2.2%) | 60 (−33%) |
| turning on the spot (one block) | 20 | **20** | 8 |

- Scripted blocks: the ensemble made **0 errors in 12 blocks**.
- The 90% intervals contained the true count in **16 of 16** test blocks.
- Be honest in the caption: the advantage on hard movements is large but rests on only 4 test blocks, so it is not yet statistically conclusive.

End on a large live-style counter showing a real test result, S009's normal-pace walk: **"40 steps, 90% interval 38–42"** (estimate 40 ± 2.1; I really took 40). Add a recap row of icons: sensor → windows → CNN → probability peaks → × 2 → count ± interval.

## Training data, for context (show it small somewhere)

- 10 recorded sessions, about 5 minutes each.
- 8 sessions for training (578 labeled strides); 2 kept locked for the final test (179 strides).
- Leave-one-session-out and 4-fold cross-validation, so no model was ever scored on a session it trained on.
