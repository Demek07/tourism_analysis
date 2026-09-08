import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# ============================================================
# 1. ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ EXCEL
# ============================================================

def load_tourism_data(file_path):
    """
    Загружает и объединяет все месячные данные из Excel-файла Росстата
    """
    monthly_sheets = ['3.2022', '3.2023', '3.2024', '3.2025']
    
    all_data = []
    
    for sheet in monthly_sheets:
        print(f"  Загрузка листа {sheet}...")
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)
        
        # Ищем строку с "Российская Федерация"
        try:
            start_row = df[df[0] == 'Российская Федерация'].index[0]
        except IndexError:
            print(f"    Не найден регион 'Российская Федерация' на листе {sheet}")
            continue
        
        df_data = pd.read_excel(
            file_path, 
            sheet_name=sheet, 
            header=None,
            skiprows=start_row
        )
        
        regions = df_data[0].values
        
        months_row = start_row - 1
        months_df = pd.read_excel(
            file_path, 
            sheet_name=sheet, 
            header=None,
            skiprows=months_row,
            nrows=1
        )
        
        months = [str(m).strip() for m in months_df.iloc[0, 1:].values if pd.notna(m)]
        data_values = df_data.iloc[:, 1:].values
        
        # Обрезаем до количества месяцев
        if len(months) < data_values.shape[1]:
            data_values = data_values[:, :len(months)]
        elif len(months) > data_values.shape[1]:
            months = months[:data_values.shape[1]]
        
        df_monthly = pd.DataFrame(data_values, columns=months)
        df_monthly['Регион'] = regions[:len(df_monthly)]
        df_monthly['Год'] = int(sheet.split('.')[1])
        df_monthly = df_monthly.dropna(subset=['Регион'])
        
        df_melted = df_monthly.melt(
            id_vars=['Регион', 'Год'],
            var_name='Месяц',
            value_name='Поездки'
        )
        
        month_map = {
            'январь': 1, 'январь-февраль': 2, 'январь-март': 3,
            'январь-апрель': 4, 'январь-май': 5, 'январь-июнь': 6,
            'январь-июль': 7, 'январь-август': 8, 'январь-сентябрь': 9,
            'январь-октябрь*': 10, 'январь-ноябрь*': 11, 'январь-декабрь*': 12
        }
        df_melted['Месяц_номер'] = df_melted['Месяц'].map(month_map)
        
        # Убираем строки с NaN в месяце
        df_melted = df_melted.dropna(subset=['Месяц_номер'])
        
        # Преобразуем в целые числа
        df_melted['Месяц_номер'] = df_melted['Месяц_номер'].astype(int)
        df_melted['Год'] = df_melted['Год'].astype(int)
        
        # Создаем дату
        df_melted['Дата'] = pd.to_datetime(
            df_melted['Год'].astype(str) + '-' + df_melted['Месяц_номер'].astype(str) + '-01'
        )
        
        all_data.append(df_melted)
    
    if not all_data:
        raise ValueError("Не удалось загрузить данные ни с одного листа!")
    
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.dropna(subset=['Поездки'])
    df_all['Поездки'] = pd.to_numeric(df_all['Поездки'], errors='coerce')
    df_all = df_all[~df_all['Регион'].str.contains('К содержанию|NaN', na=False)]
    df_all = df_all[df_all['Регион'] != '']
    
    # Убираем строки, где регион - это федеральные округа
    federal_districts = [
        'Центральный федеральный округ',
        'Северо-Западный федеральный округ',
        'Южный федеральный округ',
        'Северо-Кавказский федеральный округ',
        'Приволжский федеральный округ',
        'Уральский федеральный округ',
        'Сибирский федеральный округ',
        'Дальневосточный федеральный округ'
    ]
    df_all = df_all[~df_all['Регион'].isin(federal_districts)]
    
    print(f"  Загружено {len(df_all)} записей")
    return df_all


# ============================================================
# 2. ОПРЕДЕЛЕНИЕ РЕГИОНОВ
# ============================================================

def get_regions(df_all, top_n=None):
    """
    Возвращает список регионов для анализа
    """
    df_2025 = df_all[df_all['Год'] == 2025].copy()
    region_total = df_2025.groupby('Регион')['Поездки'].sum().sort_values(ascending=False)
    
    if top_n is None:
        regions = region_total.index.tolist()
        print(f"\n{'='*60}")
        print(f"АНАЛИЗ ВСЕХ РЕГИОНОВ (всего {len(regions)} регионов)")
        print(f"{'='*60}")
    else:
        regions = region_total.head(top_n).index.tolist()
        print(f"\n{'='*60}")
        print(f"ТОП-{top_n} РЕГИОНОВ ПО ТУРПОТОКУ ЗА 2025 ГОД")
        print(f"{'='*60}")
    
    for i, region in enumerate(regions[:20], 1):
        total = region_total[region]
        print(f"{i:2d}. {region:<35} {total:>12,.0f} поездок")
    
    if len(regions) > 20:
        print(f"... и еще {len(regions)-20} регионов")
    
    return regions


# ============================================================
# 3. ФИЛЬТРАЦИЯ
# ============================================================

def filter_regions_data(df_all, regions):
    """
    Фильтрует данные для указанных регионов
    """
    df_filtered = df_all[df_all['Регион'].isin(regions)].copy()
    df_filtered = df_filtered.sort_values(['Регион', 'Дата'])
    df_filtered['Дата'] = pd.to_datetime(df_filtered['Дата'])
    
    return df_filtered


# ============================================================
# 4. ОЧИСТКА ДАННЫХ (ИСПРАВЛЕННАЯ)
# ============================================================

def clean_data(df_filtered):
    """
    Очистка данных: обработка пропусков, выбросов, создание дополнительных признаков
    """
    df_clean = df_filtered.copy()
    
    # Проверка пропусков
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        print(f"\nПропуски в данных:\n{missing[missing > 0]}")
    
    # ИСПРАВЛЕНО: Интерполяция через группировку с линейным методом
    def interpolate_group(group):
        """Интерполяция для каждой группы"""
        return group.interpolate(method='linear', limit_direction='both')
    
    # Сбрасываем индекс, группируем и интерполируем
    df_clean = df_clean.sort_values(['Регион', 'Дата'])
    
    # Интерполяция по каждой группе
    df_clean['Поездки'] = df_clean.groupby('Регион')['Поездки'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # Если остались пропуски (например, весь ряд пустой), заполняем нулями
    df_clean['Поездки'] = df_clean['Поездки'].fillna(0)
    
    # Выбросы (IQR метод)
    def detect_outliers(group):
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (group < lower_bound) | (group > upper_bound)
    
    df_clean['is_outlier'] = df_clean.groupby('Регион')['Поездки'].transform(detect_outliers)
    
    # Feature Engineering
    df_clean = df_clean.sort_values(['Регион', 'Дата'])
    
    # Лаговые значения
    for lag in [1, 3, 6, 12]:
        df_clean[f'lag_{lag}'] = df_clean.groupby('Регион')['Поездки'].shift(lag)
    
    # Скользящие средние
    for window in [3, 6, 12]:
        df_clean[f'ma_{window}'] = df_clean.groupby('Регион')['Поездки'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # Календарные признаки
    df_clean['месяц'] = df_clean['Дата'].dt.month
    df_clean['квартал'] = df_clean['Дата'].dt.quarter
    df_clean['год'] = df_clean['Дата'].dt.year
    df_clean['день_года'] = df_clean['Дата'].dt.dayofyear
    
    # Признаки праздников
    holiday_months = [1, 2, 3, 5, 12]
    df_clean['is_holiday_month'] = df_clean['месяц'].isin(holiday_months).astype(int)
    
    # Сезон
    df_clean['season'] = df_clean['месяц'].apply(lambda m: 
        'зима' if m in [12, 1, 2] else
        'весна' if m in [3, 4, 5] else
        'лето' if m in [6, 7, 8] else 'осень'
    )
    
    print(f"\nРазмер данных после очистки: {len(df_clean)} записей")
    print(f"Обнаружено выбросов: {df_clean['is_outlier'].sum()}")
    
    return df_clean


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_trends_and_seasonality(df_clean, top_n=10):
    """
    Визуализация трендов и сезонности
    """
    top_regions = df_clean.groupby('Регион')['Поездки'].sum().nlargest(top_n).index
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Тренды
    for region in top_regions:
        data = df_clean[df_clean['Регион'] == region]
        axes[0].plot(data['Дата'], data['Поездки'], label=region, linewidth=2)
    
    axes[0].set_title(f'Динамика турпотока по ТОП-{top_n} регионам (2022-2025)', fontsize=14)
    axes[0].set_xlabel('Дата')
    axes[0].set_ylabel('Количество поездок')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True)
    
    # Сезонность
    seasonality = df_clean.groupby(['Регион', 'месяц'])['Поездки'].mean().reset_index()
    
    for region in top_regions:
        data = seasonality[seasonality['Регион'] == region]
        axes[1].plot(data['месяц'], data['Поездки'], marker='o', label=region)
    
    axes[1].set_title('Сезонная компонента (среднее по месяцам)', fontsize=14)
    axes[1].set_xlabel('Месяц')
    axes[1].set_ylabel('Среднее количество поездок')
    axes[1].set_xticks(range(1, 13))
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_regional_comparison(df_clean, top_n=10):
    """
    Сравнительный анализ регионов
    """
    top_regions = df_clean.groupby('Регион')['Поездки'].sum().nlargest(top_n).index
    df_top = df_clean[df_clean['Регион'].isin(top_regions)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Общий объем
    total_by_region = df_top.groupby('Регион')['Поездки'].sum().sort_values()
    total_by_region.plot(kind='barh', ax=axes[0, 0], color=sns.color_palette("viridis", len(total_by_region)))
    axes[0, 0].set_title('Общий объем турпотока по регионам', fontsize=12)
    axes[0, 0].set_xlabel('Всего поездок')
    
    # 2. Рост
    growth = df_top.groupby('Регион').apply(
        lambda x: (x[x['Год'] == 2025]['Поездки'].mean() / x[x['Год'] == 2022]['Поездки'].mean() - 1) * 100
    )
    growth = growth.sort_values()
    growth.plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Рост турпотока 2022→2025, %', fontsize=12)
    axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.3)
    
    # 3. Распределение
    df_top.boxplot(column='Поездки', by='Регион', ax=axes[1, 0])
    axes[1, 0].set_title('Распределение месячного турпотока', fontsize=12)
    axes[1, 0].set_xlabel('Регион')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Корреляция
    pivot_regions = df_top.pivot_table(
        values='Поездки', 
        index='Дата', 
        columns='Регион'
    )
    corr_matrix = pivot_regions.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', ax=axes[1, 1], cbar_kws={'label': 'Корреляция'})
    axes[1, 1].set_title('Корреляция между регионами', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return pivot_regions


def plot_all_regions_summary(df_clean):
    """
    Сводный график для всех регионов
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Общий тренд по России
    russia_total = df_clean.groupby('Дата')['Поездки'].sum().reset_index()
    axes[0, 0].plot(russia_total['Дата'], russia_total['Поездки'], linewidth=2, color='darkblue')
    axes[0, 0].set_title('Общий турпоток по России (все регионы)', fontsize=14)
    axes[0, 0].set_xlabel('Дата')
    axes[0, 0].set_ylabel('Всего поездок')
    axes[0, 0].grid(True)
    
    # 2. Распределение регионов по объему
    totals = df_clean.groupby('Регион')['Поездки'].sum().sort_values(ascending=False)
    totals.head(20).plot(kind='bar', ax=axes[0, 1], color='teal')
    axes[0, 1].set_title('ТОП-20 регионов по общему турпотоку', fontsize=14)
    axes[0, 1].set_xlabel('Регион')
    axes[0, 1].set_ylabel('Всего поездок')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Распределение долей
    top10 = totals.head(10)
    others = pd.Series({'Остальные': totals.tail(len(totals)-10).sum()})
    pie_data = pd.concat([top10, others])
    axes[1, 0].pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Доли регионов в общем турпотоке', fontsize=14)
    
    # 4. Средняя сезонность по России
    monthly_avg = df_clean.groupby('месяц')['Поездки'].mean().reset_index()
    axes[1, 1].bar(monthly_avg['месяц'], monthly_avg['Поездки'], color='coral')
    axes[1, 1].set_title('Среднемесячный турпоток по России', fontsize=14)
    axes[1, 1].set_xlabel('Месяц')
    axes[1, 1].set_ylabel('Среднее количество поездок')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_decomposition(df_clean, region):
    """
    Декомпозиция временного ряда для региона
    """
    data = df_clean[df_clean['Регион'] == region].copy()
    if len(data) < 24:
        print(f"  Недостаточно данных для декомпозиции {region}")
        return
    
    data = data.set_index('Дата').sort_index()
    
    try:
        decomposition = seasonal_decompose(data['Поездки'], model='additive', period=12)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition.observed.plot(ax=axes[0], title=f'Исходный ряд - {region}')
        decomposition.trend.plot(ax=axes[1], title='Тренд')
        decomposition.seasonal.plot(ax=axes[2], title='Сезонность')
        decomposition.resid.plot(ax=axes[3], title='Остатки')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"  Ошибка декомпозиции для {region}: {e}")


# ============================================================
# 6. СТАТИСТИКА
# ============================================================

def get_region_statistics(df_clean, top_n=None):
    """
    Выводит статистику по регионам
    """
    if top_n is not None:
        top_regions = df_clean.groupby('Регион')['Поездки'].sum().nlargest(top_n).index
        df_stats = df_clean[df_clean['Регион'].isin(top_regions)]
    else:
        df_stats = df_clean
    
    stats = df_stats.groupby('Регион').agg({
        'Поездки': ['mean', 'std', 'min', 'max', 'count'],
        'Год': lambda x: x.unique().tolist()
    })
    stats.columns = ['Среднее', 'Стд_откл', 'Мин', 'Макс', 'Кол-во', 'Годы']
    stats = stats.sort_values(('Среднее'), ascending=False)
    
    print("\n" + "="*60)
    print("СТАТИСТИКА ПО РЕГИОНАМ")
    print("="*60)
    print(stats.round(0).head(20))
    
    return stats


def seasonal_analysis(df_clean, top_n=None):
    """
    Детальный анализ сезонности
    """
    if top_n is not None:
        top_regions = df_clean.groupby('Регион')['Поездки'].sum().nlargest(top_n).index
        df_analysis = df_clean[df_clean['Регион'].isin(top_regions)]
    else:
        df_analysis = df_clean
    
    monthly_avg = df_analysis.groupby(['Регион', 'месяц'])['Поездки'].mean().reset_index()
    peak_months = monthly_avg.loc[
        monthly_avg.groupby('Регион')['Поездки'].idxmax()
    ][['Регион', 'месяц']].set_index('Регион')
    
    print("\n" + "="*60)
    print("СЕЗОННЫЕ ПИКИ ПО РЕГИОНАМ")
    print("="*60)
    month_names = {1:'Янв',2:'Фев',3:'Мар',4:'Апр',5:'Май',6:'Июн',
                   7:'Июл',8:'Авг',9:'Сен',10:'Окт',11:'Ноя',12:'Дек'}
    for region, month in peak_months.head(20).iterrows():
        print(f"{region[:35]:<35} → пик в {month_names[month['месяц']]}")
    
    summer_avg = df_analysis[df_analysis['месяц'].isin([6,7,8])].groupby('Регион')['Поездки'].mean()
    winter_avg = df_analysis[df_analysis['месяц'].isin([12,1,2])].groupby('Регион')['Поездки'].mean()
    ratio = (summer_avg / winter_avg).sort_values(ascending=False)
    
    print("\n" + "="*60)
    print("СООТНОШЕНИЕ ЛЕТО/ЗИМА")
    print("="*60)
    for region, r in ratio.dropna().head(10).items():
        print(f"{region[:35]:<35} → летом в {r:.1f}x больше туристов, чем зимой")


# ============================================================
# 7. ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================

def run_full_analysis(file_path, top_n=None):
    """
    Запускает полный анализ
    """
    print("="*60)
    print("АНАЛИЗ ТУРИСТИЧЕСКОГО ПОТОКА РОССИИ")
    print("="*60)
    
    # 1. Загрузка
    print("\n[1/6] Загрузка данных...")
    df_all = load_tourism_data(file_path)
    print(f"Загружено {len(df_all)} записей за 2022-2025 гг.")
    
    # 2. Определение регионов
    print("\n[2/6] Определение регионов для анализа...")
    regions = get_regions(df_all, top_n)
    
    # 3. Фильтрация
    print("\n[3/6] Фильтрация данных...")
    df_filtered = filter_regions_data(df_all, regions)
    print(f"Отфильтровано {len(df_filtered)} записей")
    
    # 4. Очистка
    print("\n[4/6] Очистка данных...")
    df_clean = clean_data(df_filtered)
    
    # 5. Статистика
    print("\n[5/6] Расчет статистики...")
    stats = get_region_statistics(df_clean, top_n=min(20, len(df_clean['Регион'].unique())))
    
    # 6. Визуализация
    print("\n[6/6] Визуализация...")
    
    if top_n is not None:
        plot_trends_and_seasonality(df_clean, top_n)
        plot_regional_comparison(df_clean, top_n)
        if len(regions) > 0:
            plot_decomposition(df_clean, regions[0])
    else:
        plot_all_regions_summary(df_clean)
        plot_trends_and_seasonality(df_clean, top_n=10)
        plot_regional_comparison(df_clean, top_n=10)
        if len(regions) > 0:
            plot_decomposition(df_clean, regions[0])
    
    seasonal_analysis(df_clean, top_n=min(20, len(df_clean['Регион'].unique())))
    
    # Сохранение
    output_file = f'tourism_clean_{"all" if top_n is None else f"top{top_n}"}.csv'
    df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nДанные сохранены в {output_file}")
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("="*60)
    
    return df_clean, regions, stats


# ============================================================
# 8. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    FILE_PATH = 'Turpotok_1kv-2026.xlsx'
    
    # Выберите режим:
    # top_n=None — все регионы
    # top_n=10 — только ТОП-10
    
    df_clean, regions, stats = run_full_analysis(FILE_PATH, top_n=None)
    # df_clean, regions, stats = run_full_analysis(FILE_PATH, top_n=10)