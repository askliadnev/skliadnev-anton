"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore")


df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_random_seed(42)

CATEGORY_MAPPING = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
category_to_id = {cat: idx for idx, cat in enumerate(CATEGORY_MAPPING)}

def compute_snips_metric(probabilities, actions, rewards, baseline_prob=1/3):
    with torch.no_grad():
        weights = probabilities[torch.arange(len(actions)), actions] / baseline_prob
        denominator = weights.sum().clamp_min(1e-12)
        numerator = (weights * rewards).sum()
        return (numerator / denominator).item()

class PolicyNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 3)
        with torch.no_grad():
            nn.init.normal_(self.output_layer.weight, std=0.01)
            self.output_layer.bias.data = torch.tensor([2.0, -1.0, -1.0])
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        logits = self.output_layer(x)
        logits[:, 0] += 2.0
        return logits

continuous_features = ["recency", "history"]
binary_features = ["mens", "womens", "newbie"] 
categorical_features = ["zip_code", "channel", "history_segment"]
all_features = continuous_features + binary_features + categorical_features

unique_categories = df_train['segment'].unique()
invalid_categories = set(unique_categories) - set(CATEGORY_MAPPING)
if invalid_categories:
    df_train['segment'] = df_train['segment'].apply(
        lambda x: x if x in CATEGORY_MAPPING else "No E-Mail")

X_data = df_train[all_features].copy()
X_test_data = df_test[all_features].copy()

preprocessor = ColumnTransformer(
    transformers=[
        ("continuous", StandardScaler(), continuous_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("binary", "passthrough", binary_features),
    ],
    remainder="drop",
)

X_processed = preprocessor.fit_transform(X_data)
X_test_processed = preprocessor.transform(X_test_data)
X_processed = X_processed.astype("float32")
X_test_processed = X_test_processed.astype("float32")
actions = df_train["segment"].map(category_to_id).astype(int).values
rewards = df_train["visit"].astype(int).values

action_performance = df_train.groupby('segment').agg({
    'visit': ['count', 'mean', 'sum']
}).round(4)
action_performance.columns = ['count', 'conversion_rate', 'total_visits']
best_action_rate = action_performance.loc['Mens E-Mail', 'conversion_rate']

X_tensor = torch.tensor(X_processed, dtype=torch.float32)
rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
actions_tensor = torch.tensor(actions, dtype=torch.long)

policy_model = PolicyNetwork(X_processed.shape[1])
model_optimizer = optim.Adam(policy_model.parameters(), lr=0.002, weight_decay=0.05)

training_epochs = 150
best_metric = -float('inf')
best_weights = None

policy_model.train()
for epoch in range(training_epochs):
    model_optimizer.zero_grad()
    output_logits = policy_model(X_tensor)
    action_probabilities = torch.softmax(output_logits, dim=1)
    importance_weights = (rewards_tensor / (1/3)).clamp_max(5.0)
    selected_action_probs = action_probabilities[torch.arange(len(actions_tensor)), actions_tensor]
    policy_loss = -(importance_weights * torch.log(selected_action_probs + 1e-8)).mean()
    target_distribution = torch.tensor([0.95, 0.03, 0.02]).repeat(action_probabilities.shape[0], 1)
    regularization_term = torch.nn.functional.kl_div(
        torch.log(action_probabilities + 1e-8), target_distribution, reduction='batchmean'
    )
    total_loss = policy_loss + 5.0 * regularization_term
    total_loss.backward()
    model_optimizer.step()
    
    if epoch % 25 == 0 or epoch == training_epochs - 1:
        with torch.no_grad():
            snips_metric = compute_snips_metric(action_probabilities, actions_tensor, rewards_tensor)
            performance_score = snips_metric - best_action_rate
            if performance_score > best_metric:
                best_metric = performance_score
                best_weights = policy_model.state_dict().copy()
            print(f"Эпоха {epoch+1:3d}: Loss = {total_loss.item():.4f}")


policy_model.load_state_dict(best_weights)
policy_model.eval()


with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    test_logits = policy_model(X_test_tensor)
    test_probabilities = torch.softmax(test_logits, dim=1)
    test_probs_numpy = test_probabilities.numpy()


submission = pd.DataFrame({
    "p_mens_email": test_probs_numpy[:, 0],
    "p_womens_email": test_probs_numpy[:, 1],
    "p_no_email": test_probs_numpy[:, 2],
    "id": df_test["id"].values,
})


assert np.all(np.isfinite(submission[["p_mens_email", "p_womens_email", "p_no_email"]].values))
assert np.allclose(
    submission[["p_mens_email", "p_womens_email", "p_no_email"]].sum(axis=1),
    1.0,
    atol=1e-6,
)

def create_submission(submission_df):
    """
    Создание файла submission.csv в папку results
    """
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path

def main():
    """
    Главная функция программы
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)

if __name__ == "__main__":
    main()
