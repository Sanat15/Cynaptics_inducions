For the last task , i tried using both gpt2 and llama-2-7b-hf for the base model.
 * For this task it seamed that llama performeb better so my final choice of model was llama.
 * For fine tuning this model , my key stratigies were --
     * Minimize the NLL (causal language modeling).
     *  Warm-up + weight decay for stabilization and regularization.
     *  Evaluate and save the model at the end of every epoch.
     *  Optimized for GPU memory usage and speed.
     *  Efficient batch processing for varying sequence lengths.
 * The reason for using these stratigies are
     * Alignment with the Objective (Causal Language Modeling)
     * Efficient Use of Resources (Mixed Precision and Dynamic Padding)
     * Robust Training Configuration
     * Evaluation-Centric Approach
     * Scalable and Maintainable Implementation
