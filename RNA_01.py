# Código em PyTorch – Projeto PMC com 5 treinamentos e análise completa
# Requisitos: torch, pandas, matplotlib, openpyxl, scikit-learn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# === Leitura dos dados ===
file_path = r"C:\Users\Lucas\Desktop\DadosProjeto01RNA.xlsx"
df_train = pd.read_excel(file_path, sheet_name="DadosTreinamentoRNA")
df_test = pd.read_excel(file_path, sheet_name="DadosTesteRNA")
df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

X_train = df_train[['x1', 'x2', 'x3']].values.astype(np.float32)
T_train = df_train[['d']].values.astype(np.float32)
X_test = df_test[['x1', 'x2', 'x3']].values.astype(np.float32)
T_test = df_test[['d']].values.astype(np.float32)

X_train_tensor = torch.tensor(X_train)
T_train_tensor = torch.tensor(T_train)
X_test_tensor = torch.tensor(X_test)
T_test_tensor = torch.tensor(T_test)

# === Modelo ===
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 10)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 1)
        self.act2 = nn.Sigmoid()
    def forward(self, x):
        x = self.act1(self.fc1(x))
        return self.act2(self.fc2(x))

resultados = []
erros_relativos = []
tabela_epocas = []
tabela_mse = []
erros_por_epoca = {}

# === Treinamentos T1 a T5 ===
for i in range(5):
    torch.manual_seed(i + 1)
    model = NeuralNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_history = []
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, T_train_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if loss.item() < 1e-6:
            break

    erros_por_epoca[f"T{i+1}"] = loss_history
    tabela_epocas.append(epoch + 1)
    tabela_mse.append(loss.item())

    model.eval()
    with torch.no_grad():
        Y_test = model(X_test_tensor).numpy().flatten()
        resultados.append(Y_test)
        erro_pct = 100 * np.abs((T_test.flatten() - Y_test) / T_test.flatten())
        erros_relativos.append(erro_pct)

# === Tabela 1 ===
df_tabela1 = pd.DataFrame({
    "Treinamento": [f"{i+1}º (T{i+1})" for i in range(5)],
    "Erro Quadrático Médio": tabela_mse,
    "Número Total de Épocas": tabela_epocas
})
print("\nTabela 1:")
print(df_tabela1)

# === Tabela 2 ===
df_resultado = df_test[['x1', 'x2', 'x3']].copy()
df_resultado['d'] = T_test.flatten()
for i in range(5):
    df_resultado[f'y (T{i+1})'] = resultados[i]
print("\nTabela 2 (parcial):")
print(df_resultado.head())

# === Erro relativo médio e variância ===
erros_df = pd.DataFrame(erros_relativos).T
media_erro = erros_df.mean().values
variancia_erro = erros_df.var().values

df_summary = pd.DataFrame({
    'Treinamento': [f"T{i+1}" for i in range(5)],
    'Erro relativo médio (%)': media_erro,
    'Variância (%)': variancia_erro
})
print("\nResumo dos erros:")
print(df_summary)

# === Gráfico: 2 treinamentos com mais épocas ===
top2_idx = np.argsort(tabela_epocas)[-2:]
plt.figure(figsize=(10, 5))
for idx in top2_idx:
    plt.plot(erros_por_epoca[f"T{idx+1}"], label=f"T{idx+1}")
plt.title("Erro Quadrático Médio vs Épocas (Top 2 Treinamentos)")
plt.xlabel("Épocas")
plt.ylabel("Erro Quadrático Médio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
