import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# ЗАГРУЗКА И ПЕРВИЧНАЯ ОБРАБОТКА ДАННЫХ
# ========================================

try:
    df = pd.read_excel('salesworkload.xlsx', sheet_name='sales_figures', header=1)
    logger.info(f"Данные успешно загружены: {len(df)} строк, {len(df.columns)} столбцов")
except FileNotFoundError:
    logger.error("Файл 'salesworkload.xlsx' не найден")
    exit()
except ValueError as e:
    logger.error(f"Лист 'sales_figures' не найден: {e}")
    exit()

# Удаление столбца Customer
df = df.drop(columns=['Customer'], errors='ignore')

# Преобразование числовых столбцов
numeric_cols = ['HoursOwn', 'HoursLease', 'Sales units', 'Turnover', 'Area (m2)']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Удаление строк с пропусками в ключевых столбцах
df.dropna(subset=['MonthYear', 'Turnover', 'HoursOwn', 'HoursLease'], inplace=True)
logger.info(f"После удаления пропусков осталось {len(df)} строк")

# ========================================
# СОЗДАНИЕ ПРОИЗВОДНЫХ ПРИЗНАКОВ
# ========================================

# Расчет общей рабочей нагрузки (Workload Index)
df['TotalHours'] = df['HoursOwn'] + df['HoursLease']

# Извлечение временных признаков
df['Date'] = pd.to_datetime(df['MonthYear'], format='%m.%Y', errors='coerce')
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# ========================================
# РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)
# ========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
sns.set_theme(style="darkgrid")
fig.suptitle('Sales and Workload Analysis', fontsize=20, fontweight='bold', y=0.995)

# Scatter plot: Связь между рабочей нагрузкой и продажами
ax1 = axes[0, 0]
correlation = df['TotalHours'].corr(df['Turnover'])
sns.scatterplot(data=df, x='TotalHours', y='Turnover', ax=ax1,
                alpha=0.4, color='#F47A9A', s=30, edgecolor=None)
ax1.set_title('Sales vs Workload Index', fontsize=14, fontweight='bold')
ax1.set_xlabel('Workload Index (Total Hours)', fontsize=12)
ax1.set_ylabel('Sales (Turnover, EUR)', fontsize=12)
ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
         transform=ax1.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
ax1.set_xlim(left=-500)

# Histogram: Распределение продаж
ax2 = axes[0, 1]
sns.histplot(data=df, x='Turnover', ax=ax2, color='#F47A9A',
             bins=30, edgecolor='black', alpha=0.7)
ax2.set_title('Distribution of Sales', fontsize=14, fontweight='bold')
ax2.set_xlabel('Sales (EUR)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)

# Histogram: Распределение рабочей нагрузки
ax3 = axes[1, 0]
sns.histplot(data=df, x='TotalHours', ax=ax3, color='lightsalmon',
             bins=40, edgecolor='black', alpha=0.8)
ax3.set_title('Distribution of Workload Index', fontsize=14, fontweight='bold')
ax3.set_xlabel('Workload Index (Total Hours)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)

# Bar plot: Средние продажи по отделам
ax4 = axes[1, 1]
if 'Dept. Name' in df.columns:
    avg_sales_dept = df.groupby('Dept. Name')['Turnover'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_sales_dept.values, y=avg_sales_dept.index,
                ax=ax4, color='steelblue', orient='h')
    ax4.set_title('Average Sales by Department', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Average Sales (EUR)', fontsize=12)
    ax4.set_ylabel('')

plt.tight_layout()
plt.show()

# ========================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИРОВАНИЯ
# ========================================

# One-hot encoding категориальных признаков
categorical_cols = ['StoreID', 'Dept_ID', 'Dept. Name', 'Country', 'City', 'Opening hours']
categorical_cols = [col for col in categorical_cols if col in df.columns]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Удаление оставшихся пропусков
initial_rows = len(df_encoded)
df_encoded.dropna(inplace=True)
final_rows = len(df_encoded)
logger.info(f"Удалено строк с пропусками: {initial_rows - final_rows}, осталось: {final_rows}")

# Разделение на признаки (X) и целевую переменную (y)
excluded_cols = ['Turnover', 'MonthYear', 'Date', 'HoursOwn', 'HoursLease', 'Sales units']
features = [col for col in df_encoded.columns if col not in excluded_cols]
target = 'Turnover'

X = df_encoded[features]
y = df_encoded[target]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logger.info(f"Обучающая выборка: {len(X_train)} строк, тестовая: {len(X_test)} строк")

# ========================================
# ПОСТРОЕНИЕ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ
# ========================================

# Линейная регрессия
logger.info("\n--- Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)
logger.info(f"RMSE: {lr_rmse:.2f} EUR")
logger.info(f"R² Score: {lr_r2:.4f}")

# Random Forest
logger.info("\n--- Random Forest Regressor ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)
logger.info(f"RMSE: {rf_rmse:.2f} EUR")
logger.info(f"R² Score: {rf_r2:.4f}")

# ========================================
# АНАЛИЗ РЕЗУЛЬТАТОВ И ВЫВОДЫ
# ========================================

# Проверка гипотезы о влиянии рабочей нагрузки на продажи
if 'TotalHours' in X_train.columns:
    workload_coef_index = list(X_train.columns).index('TotalHours')
    workload_coef = lr_model.coef_[workload_coef_index]

    logger.info(f"\n--- Анализ влияния рабочей нагрузки ---")
    logger.info(f"Коэффициент 'TotalHours' в Linear Regression: {workload_coef:.2f}")

    if workload_coef > 0:
        logger.info("✓ Прямая зависимость: увеличение рабочей нагрузки связано с ростом продаж")
    else:
        logger.info("✓ Обратная зависимость: увеличение рабочей нагрузки связано со снижением продаж")

# Важность признаков (Feature Importance) из Random Forest
logger.info("\n--- Топ-10 наиболее важных факторов для продаж ---")
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importances.head(10).to_string(index=False))

# ========================================
# ПРОГНОЗИРОВАНИЕ ПРИ ОПТИМИЗАЦИИ НАГРУЗКИ
# ========================================

# Симуляция увеличения рабочей нагрузки на 10%
if 'TotalHours' in X_test.columns:
    X_test_optimized = X_test.copy()
    X_test_optimized['TotalHours'] = X_test_optimized['TotalHours'] * 1.10

    optimized_preds = rf_model.predict(X_test_optimized)

    # Расчет ожидаемого роста продаж
    original_avg_sales = y_test.mean()
    optimized_avg_sales = optimized_preds.mean()
    sales_increase_percent = ((optimized_avg_sales - original_avg_sales) / original_avg_sales) * 100

    logger.info(f"\n--- Прогноз при увеличении нагрузки на 10% ---")
    logger.info(f"Текущие средние продажи: {original_avg_sales:.2f} EUR")
    logger.info(f"Прогнозируемые продажи: {optimized_avg_sales:.2f} EUR")
    logger.info(f"Ожидаемое изменение продаж: {sales_increase_percent:+.2f}%")