import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_excel('salesworkload.xlsx', sheet_name='sales_figures', header=1)
    logger.info("Данные успешно загружены.")
except FileNotFoundError:
    logger.warning("Ошибка: Файл 'salesworkload.xlsx' не найден.")
    exit()
except ValueError as e:
    logger.warning(f"Ошибка: Лист 'sales_figures' не найден в файле. {e}")
    exit()

df = df.drop(columns=['Customer'], errors='ignore')
numeric_cols = ['HoursOwn', 'HoursLease', 'Sales units', 'Turnover', 'Area (m2)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['MonthYear', 'Turnover', 'HoursOwn', 'HoursLease'], inplace=True)

df['TotalHours'] = df['HoursOwn'] + df['HoursLease']
df['Date'] = pd.to_datetime(df['MonthYear'], format='%m.%Y')
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
sns.set_theme(style="darkgrid")
fig.suptitle('Анализ продаж и рабочей нагрузки', fontsize=24, fontweight='bold', y=1.02)

ax1 = axes[0, 0]
correlation = df['TotalHours'].corr(df['Turnover'])
sns.scatterplot(data=df, x='TotalHours', y='Turnover', ax=ax1, alpha=0.3, color='#F47A9A', s=20, edgecolor=None)
ax1.set_title('Sales vs Workload Index', fontsize=18, fontweight='bold')
ax1.set_xlabel('Workload Index', fontsize=14)
ax1.set_ylabel('Sales (Turnover, EUR)', fontsize=14)
ax1.text(0.15, 0.95, f'Корреляция: {correlation:.3f}', transform=ax1.transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
ax1.set_xlim(left=-500)
ax2 = axes[0, 1]
sns.histplot(data=df, x='Turnover', ax=ax2, color='#F47A9A', bins=30, edgecolor='black', alpha=0.8)
ax2.set_title('Distribution of Sales', fontsize=18, fontweight='bold')
ax2.set_xlabel('Sales (EUR)', fontsize=14)
ax2.set_ylabel('Frequency', fontsize=14)
ax3 = axes[1, 0]
sns.histplot(data=df, x='TotalHours', ax=ax3, color='lightsalmon', bins=40, edgecolor='black', alpha=0.9)
ax3.set_title('Distribution of Workload Index', fontsize=18, fontweight='bold')
ax3.set_xlabel('Workload Index', fontsize=14)
ax3.set_ylabel('Frequency', fontsize=14)
ax4 = axes[1, 1]
avg_sales_dept = df.groupby('Dept. Name')['Turnover'].mean().sort_values(ascending=False)
sns.barplot(x=avg_sales_dept.values, y=avg_sales_dept.index, ax=ax4, color='steelblue', orient='h')
ax4.set_title('Average Sales by Department', fontsize=18, fontweight='bold')
ax4.set_xlabel('Average Sales (EUR)', fontsize=14)
ax4.set_ylabel('')
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

categorical_cols = ['StoreID', 'Dept_ID', 'Dept. Name', 'Country', 'City', 'Opening hours']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
initial_rows = len(df_encoded)
df_encoded.dropna(inplace=True)
final_rows = len(df_encoded)
logger.info(f"Удалено {initial_rows - final_rows} строк с пропусками. Осталось {final_rows} строк.")

excluded_cols = ['Turnover', 'MonthYear', 'Date', 'HoursOwn', 'HoursLease', 'Sales units']
features = [col for col in df_encoded.columns if col not in excluded_cols]
target = 'Turnover'

X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Данные разделены: {len(X_train)} строк в обучающей выборке, {len(X_test)} в тестовой.")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

logger.info(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, lr_preds)):.2f}")
logger.info(f"коэффициент детерминации: {r2_score(y_test, lr_preds):.4f}")
logger.info(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, rf_preds)):.2f}")
logger.info(f"коэффициент детерминации: {r2_score(y_test, rf_preds):.4f}")

workload_coef_index = list(X_train.columns).index('TotalHours')
workload_coef = lr_model.coef_[workload_coef_index]
logger.info(f"Коэффициент для 'TotalHours' в LinearRegression: {workload_coef:.2f}")
if workload_coef > 0:
    logger.info("Гипотеза подтвердилась: зависимость прямая. С увеличением рабочей нагрузки продажи растут.")
else:
    logger.info("Гипотеза подтвердилась: зависимость обратная. С увеличением рабочей нагрузки продажи падают.")

logger.info("\n4.2. Наиболее важные факторы для продаж (по мнению RandomForest):")
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
logger.info(feature_importances.head(10))

X_test_optimized = X_test.copy()
X_test_optimized['TotalHours'] = X_test_optimized['TotalHours'] * 1.10

optimized_preds = rf_model.predict(X_test_optimized)

original_avg_sales = y_test.mean()
optimized_avg_sales = optimized_preds.mean()
sales_increase_percent = ((optimized_avg_sales - original_avg_sales) / original_avg_sales) * 100

logger.info(f"Средние продажи на тестовой выборке: {original_avg_sales:.2f}")
logger.info(f"Прогнозируемые средние продажи при увеличении нагрузки на 10%: {optimized_avg_sales:.2f}")
logger.info(f"Ожидаемый рост продаж при оптимизации рабочей нагрузки на 10% составляет: {sales_increase_percent:.2f}%")