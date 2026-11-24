from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Carpeta exacta donde está guardado este archivo .py
BASE_DIR = Path(__file__).resolve().parent
RUTA_CSV = BASE_DIR / 'df_limpio.csv'

# ---------------------------------------------------------
# 1. Carga y preparación de datos
# ---------------------------------------------------------
df = pd.read_csv(RUTA_CSV)
df = df[(df['weight'] > 0) & (df['length'] > 0)].copy()

conteo_especies = df['fishcode'].value_counts()
top5_especies = conteo_especies.head(5).index.tolist()
top2_especies = conteo_especies.head(2).index.tolist()

df_top5 = df[df['fishcode'].isin(top5_especies)].copy()
df_top5['log_weight'] = np.log10(df_top5['weight'])
df_top5['log_length'] = np.log10(df_top5['length'])

# ---------------------------------------------------------
# Gráfica 1: Normalidad (Original vs Log10 y Q-Q Plot)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df_top5['weight'], bins=50, kde=False, ax=axes[0], color='#0072B2')
axes[0].set_yscale('log')
axes[0].set_title('Peso Original (Escala Y Log)', fontweight='bold')
axes[0].set_xlabel('Peso (g)')
axes[0].set_ylabel('Frecuencia')

sns.histplot(df_top5['log_weight'], bins=50, kde=True, ax=axes[1], color='#009E73')
axes[1].set_title('Peso Transformado Log10', fontweight='bold')
axes[1].set_xlabel('Log10(Peso)')
axes[1].set_ylabel('Frecuencia')

st.probplot(df_top5['log_weight'], dist='norm', plot=axes[2])
axes[2].get_lines()[0].set_color('#0072B2')
axes[2].get_lines()[0].set_markersize(2)
axes[2].get_lines()[1].set_color('#D55E00')
axes[2].set_title('Gráfico Q-Q de Log10(Peso)', fontweight='bold')

plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_01_normalidad.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Gráfica 2: Prueba de Bondad de Ajuste (Chi-Cuadrada)
# ---------------------------------------------------------
alpha_chi = 0.05
fo = df_top5['fishcode'].value_counts().loc[top5_especies].values
n_chi = fo.sum()
k_chi = len(top5_especies)
gl_chi = k_chi - 1
fe = n_chi * np.full(k_chi, 1.0 / k_chi)

chi2_stat, _ = st.chisquare(f_obs=fo, f_exp=fe)
chi2_crit = st.chi2.ppf(1 - alpha_chi, df=gl_chi)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

x_pos = np.arange(k_chi)
ancho = 0.36
axes[0].bar(x_pos - ancho/2, fo, width=ancho, color='#0072B2', label='Observada (FO)')
axes[0].bar(x_pos + ancho/2, fe, width=ancho, color='#E69F00', label='Esperada Uniforme (FE)')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(top5_especies)
axes[0].set_title('Frecuencias Observadas vs. Esperadas (Top 5)', fontweight='bold')
axes[0].set_xlabel('Especie (fishcode)')
axes[0].set_ylabel('Número de capturas')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

x_c = np.linspace(0, max(chi2_crit * 2.5, 20), 1000)
y_c = st.chi2.pdf(x_c, df=gl_chi)
axes[1].plot(x_c, y_c, color='#0072B2', linewidth=2, label=f'Distribución Chi2 (gl={gl_chi})')
axes[1].fill_between(x_c, y_c, where=(x_c >= chi2_crit), color='#D55E00', alpha=0.45, label='Región de Rechazo')
axes[1].fill_between(x_c, y_c, where=(x_c < chi2_crit), color='#009E73', alpha=0.25, label='Región de No Rechazo')
axes[1].axvline(chi2_crit, color='darkred', linestyle=':', linewidth=2, label=f'Crítico = {chi2_crit:.3f}')
axes[1].set_title(f'Distribución Chi-Cuadrada (Chi2 calc. = {chi2_stat:.1f})', fontweight='bold')
axes[1].set_xlabel('Estadístico Chi2')
axes[1].set_ylabel('Densidad')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_02_chi2.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Gráfica 3: Prueba F para Dos Varianzas
# ---------------------------------------------------------
esp1, esp2 = top2_especies[0], top2_especies[1]
g1 = df_top5[df_top5['fishcode'] == esp1]['log_weight'].values
g2 = df_top5[df_top5['fishcode'] == esp2]['log_weight'].values

n1, n2 = len(g1), len(g2)
gl1_f, gl2_f = n1 - 1, n2 - 1
s1_sq = np.var(g1, ddof=1)
s2_sq = np.var(g2, ddof=1)

alpha_f = 0.05
f_stat_2var = s1_sq / s2_sq
f_crit_inf = st.f.ppf(alpha_f / 2, gl1_f, gl2_f)
f_crit_sup = st.f.ppf(1 - alpha_f / 2, gl1_f, gl2_f)

x_min_f = min(0.85, f_crit_inf * 0.95, f_stat_2var * 0.95)
x_max_f = max(1.15, f_crit_sup * 1.05, f_stat_2var * 1.05)
x_f = np.linspace(x_min_f, x_max_f, 1000)
y_f = st.f.pdf(x_f, gl1_f, gl2_f)

plt.figure(figsize=(10, 5))
plt.plot(x_f, y_f, color='#0072B2', linewidth=2, label=f'Distribución F ({gl1_f}, {gl2_f})')
plt.fill_between(x_f, y_f, where=(x_f <= f_crit_inf) | (x_f >= f_crit_sup), color='#D55E00', alpha=0.45, label='Región de Rechazo')
plt.fill_between(x_f, y_f, where=(x_f > f_crit_inf) & (x_f < f_crit_sup), color='#009E73', alpha=0.25, label='Región de No Rechazo')
plt.axvline(f_stat_2var, color='black', linestyle='--', linewidth=2, label=f'F calculado = {f_stat_2var:.4f}')
plt.axvline(f_crit_inf, color='darkred', linestyle=':', linewidth=1.5, label=f'F_inf = {f_crit_inf:.4f}')
plt.axvline(f_crit_sup, color='darkred', linestyle=':', linewidth=1.5, label=f'F_sup = {f_crit_sup:.4f}')
plt.title(f'Prueba F para Dos Varianzas ({esp1} vs {esp2})', fontweight='bold')
plt.xlabel('Estadístico F')
plt.ylabel('Densidad')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_03_prueba_f.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Gráfica 4: Prueba t de Welch
# ---------------------------------------------------------
alpha_t = 0.05
x1_bar, x2_bar = np.mean(g1), np.mean(g2)
v1, v2 = s1_sq / n1, s2_sq / n2

t_welch = (x1_bar - x2_bar) / np.sqrt(v1 + v2)
gl_welch = int(np.floor(((v1 + v2) ** 2) / ((v1 ** 2) / (n1 - 1) + (v2 ** 2) / (n2 - 1))))
t_crit_welch = st.t.ppf(1 - alpha_t / 2, df=gl_welch)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

df_top2 = df_top5[df_top5['fishcode'].isin(top2_especies)]
sns.boxplot(data=df_top2, x='fishcode', y='log_weight', order=top2_especies,
            hue='fishcode', hue_order=top2_especies, legend=False,
            palette=['#0072B2', '#E69F00'], ax=axes[0])
axes[0].set_title(f'Distribución de Log10(Peso): {esp1} vs {esp2}', fontweight='bold')
axes[0].set_xlabel('Especie (fishcode)')
axes[0].set_ylabel('Log10(Peso)')
axes[0].grid(axis='y', alpha=0.3)

lim_t = max(abs(t_welch) * 1.15, 4.5)
x_t = np.linspace(-lim_t, lim_t, 1000)
y_t = st.t.pdf(x_t, df=gl_welch)
axes[1].plot(x_t, y_t, color='#0072B2', linewidth=2, label=f'Distribución t (gl={gl_welch})')
axes[1].fill_between(x_t, y_t, where=(x_t <= -t_crit_welch) | (x_t >= t_crit_welch), color='#D55E00', alpha=0.45, label='Región de Rechazo')
axes[1].fill_between(x_t, y_t, where=(x_t > -t_crit_welch) & (x_t < t_crit_welch), color='#009E73', alpha=0.25, label='Región de No Rechazo')
axes[1].axvline(t_welch, color='black', linestyle='--', linewidth=2, label=f't obtenido = {t_welch:.2f}')
axes[1].axvline(-t_crit_welch, color='darkred', linestyle=':', linewidth=1.5)
axes[1].axvline(t_crit_welch, color='darkred', linestyle=':', linewidth=1.5)
axes[1].set_title('Prueba t de Welch (Regiones Críticas)', fontweight='bold')
axes[1].set_xlabel('Estadístico t')
axes[1].set_ylabel('Densidad')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_04_t_welch.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Gráfica 5: ANOVA de un Factor y Tukey HSD
# ---------------------------------------------------------
grupos_anova = [df_top5[df_top5['fishcode'] == esp]['log_weight'].values for esp in top5_especies]
tukey_res = st.tukey_hsd(*grupos_anova)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

paleta_okabe = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
sns.boxplot(data=df_top5, x='fishcode', y='log_weight', order=top5_especies,
            hue='fishcode', hue_order=top5_especies, legend=False,
            palette=paleta_okabe, ax=axes[0])
axes[0].set_title('Log10(Peso) por Especie (Top 5)', fontweight='bold')
axes[0].set_xlabel('Especie (fishcode)')
axes[0].set_ylabel('Log10(Peso)')
axes[0].grid(axis='y', alpha=0.3)

sns.heatmap(tukey_res.pvalue, annot=True, fmt='.2e', cmap='Blues_r',
            xticklabels=top5_especies, yticklabels=top5_especies, ax=axes[1])
axes[1].set_title('Mapa de Calor: p-values Tukey HSD', fontweight='bold')

plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_05_anova_tukey.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Gráfica 6: Modelos Predictivos de Peso y Tamaño
# ---------------------------------------------------------
df_reg = pd.get_dummies(df_top5[['fishcode', 'log_length', 'log_weight']],
                        columns=['fishcode'], drop_first=True, dtype=float)
cols_esp = [c for c in df_reg.columns if c.startswith('fishcode_')]

X_peso = df_reg[['log_length'] + cols_esp].copy()
for c in cols_esp:
    X_peso[f'inter_l_{c}'] = X_peso['log_length'] * X_peso[c]
y_peso = df_reg['log_weight']

Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_peso, y_peso, test_size=0.2, random_state=42)
modelo_peso = LinearRegression().fit(Xp_train, yp_train)
yp_pred_log = modelo_peso.predict(Xp_test)

X_tam = df_reg[['log_weight'] + cols_esp].copy()
for c in cols_esp:
    X_tam[f'inter_w_{c}'] = X_tam['log_weight'] * X_tam[c]
y_tam = df_reg['log_length']

Xt_train, Xt_test, yt_train, yt_test = train_test_split(X_tam, y_tam, test_size=0.2, random_state=42)
modelo_tamano = LinearRegression().fit(Xt_train, yt_train)
yt_pred_log = modelo_tamano.predict(Xt_test)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(yp_test, yp_pred_log, alpha=0.25, s=10, color='#0072B2')
axes[0].plot([yp_test.min(), yp_test.max()], [yp_test.min(), yp_test.max()], color='#D55E00', linestyle='--', linewidth=2)
axes[0].set_title('Modelo A: Log10(Peso) Real vs. Predicho', fontweight='bold')
axes[0].set_xlabel('Log10(Peso) Real')
axes[0].set_ylabel('Log10(Peso) Predicho')
axes[0].grid(alpha=0.3)

axes[1].scatter(yt_test, yt_pred_log, alpha=0.25, s=10, color='#009E73')
axes[1].plot([yt_test.min(), yt_test.max()], [yt_test.min(), yt_test.max()], color='#D55E00', linestyle='--', linewidth=2)
axes[1].set_title('Modelo B: Log10(Tamaño) Real vs. Predicho', fontweight='bold')
axes[1].set_xlabel('Log10(Tamaño) Real')
axes[1].set_ylabel('Log10(Tamaño) Predicho')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_DIR / 'grafica_06_modelos_predictivos.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"¡Las 6 gráficas se generaron correctamente en: {BASE_DIR}!")