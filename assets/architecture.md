# Architecture Explanation

The system uses one shared IndoBERT encoder with two prediction heads:

1. **Shared Encoder** (`indobenchmark/indobert-base-p1`)
   - Learns contextual Indonesian text representation.
2. **Sentiment Head**
   - Classifies into `negative`, `neutral`, `positive`.
3. **Toxicity Head**
   - Classifies into `non_toxic`, `toxic`.

This multitask setup is efficient for deployment because one encoder handles both tasks.
