# !pip install numpy==1.26.4
# !pip install pandas==2.2.3
# !pip install scikit-learn==1.5.2
# !pip install autogluon==1.2

import numpy as np
import pandas as pd
import warnings

from sklearn.utils.class_weight import compute_class_weight
from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings('ignore')

train_path = './train.csv'
test_path = './test.csv'
sample_submission_path = './sample_submission.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df['URL'] = train_df['URL'].str.replace(r'\[\.\]', '.', regex=True)
test_df['URL'] = test_df['URL'].str.replace(r'\[\.\]', '.', regex=True)

weights = compute_class_weight(class_weight='balanced', 
                               classes=np.unique(train_df['label']), 
                               y=train_df['label'].values)
weights /= weights.sum()
weights = list(weights)

checkpoint_names = [
    "kmack/malicious-url-detection",
    "r3ddkahili/final-complete-malicious-url-model",
    "elftsdmr/malware-url-detect"
]

predictors = []
for ckpt in checkpoint_names:
    predictor = MultiModalPredictor(label='label', problem_type='binary')
    predictor.fit(
        train_data=train_df,
        presets='best_quality',
        time_limit=None,
        column_types={'URL': 'text'},
        seed=42,
        hyperparameters={
            "model.hf_text.checkpoint_name": ckpt,
            "env.per_gpu_batch_size": 32,
            "optimization.patience": 3,
            "optimization.loss_function": "focal_loss",
            "optimization.focal_loss.alpha": weights,
        }
    )
    predictors.append(predictor)

predictions = [pred.predict_proba(test_df)[1] for pred in predictors]

final_score = predictions[0] + predictions[1] + predictions[2]

submission = pd.read_csv(sample_submission_path)
submission['probability'] = final_score
submission.to_csv('submission.csv', index=False)
