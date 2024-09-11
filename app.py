#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Исходные данные
data = [
    ['Skyeng English Old',122613,0.1577,0.6667,0.5,0.937,51124.22,23161.03,0.06,
     0.008,62429.84,0.15,9,0.2903,0.3676,0.0853,0,0,0.0181,285,3579,0.89],
    ['Skyeng English Courses & Programmes',216119,0.0427,0.7231,0.9344,0.937,116033.72,35039.72,0.10,
     0.008,130281.04,0.15,9,0.2903,0.3676,0.0853,0.00116,0.17,0.0404,5234,2937,0.6],
    ['Skysmart School',247997,0.0926,0.7593,0.9359,0.937,82208.96,15174.63,0.09,
     0.008,110584.93,0.15,9,0.2903,0.3676,0.0853,0.002432,0.13,0.0191,9143,5178,0.55],
    ['Skysmart Homeschooling',21994,0.0507,0.7415,0.9311,0.937,99869.91,14751,0.10,
     0.008,66940.63,0.15,9,0.2903,0.3676,0.0853,0.001712,0.25,0.0544,876,79,0.59],
    ['Skysmart Preschool',25746,0.0932,0.7631,0.9535,0.937,70825.75,20140.36,0.10,
     0.008,84901.29,0.15,9,0.2903,0.3676,0.0853,0.00188,0.2,0.0368,1306,357,0.59],
    ['Skysmart Coding',62258,0.0930,0.7815,0.9516,0.937,101590.19,24171.89,0.16,
     0.008,128775.13,0.15,9,0.2903,0.3676,0.0853,0.001176,0.17,0.0326,3821,862,0.5],
    ['Skysmart Exams',41348,0.1210,0.7628,0.9463,0.937,91172.79,24851.56,0.08,
     0.008,98266.09,0.15,9,0.2903,0.3676,0.0853,0.002848,0.11,0.0197,2250,603,0.76],
    ['Skysmart Exams Large Classes',19021,0.0879,0.8361,0.9391,0.937,50645.17,6888.04,0.08,
     0.008,31109.05,0.15,9,0.2903,0.3676,0.0853,0.001824,0.14,0.0356,896,179,0.65]
]

columns = ['portfolio','orders','conv','approve_rate','take_rate','CR_internal_loan',
           'ATV_bank_loan','ATV_internal_loan','bank_loan_commission','internal_loan_commission',
           'FTV_internal_loan','annual_inflation_rate','average_loan_period_months',
           'sm','cogs','ops','ya_split_comission',
           'bank_loan_refunds_share','internal_loan_refunds_share',
           'bank_payments','installment_payments','internal_loan_repayment_rate']

df = pd.DataFrame(data, columns=columns)
df['CR_bank_loan'] = df['approve_rate'] * df['take_rate']
df['months_inflation_rate'] = (1 + df['annual_inflation_rate']) ** (1/12) - 1
df['inflation_adjustment_factor'] = (1 + df['months_inflation_rate']) ** (-df['average_loan_period_months'])

# Функция расчета финансовых метрик
def calculate_financial_metrics(df, include_refunds):
    if include_refunds:
        df['cash_in_bank_loan'] = df['bank_payments'] * df['ATV_bank_loan'] * (1 + df['bank_loan_commission']) * (1 - df['bank_loan_refunds_share'])
        df['cash_in_internal'] = df['installment_payments'] * df['FTV_internal_loan'] * df['internal_loan_repayment_rate'] * df['inflation_adjustment_factor'] * (1 - df['internal_loan_refunds_share'])
        df['GMV'] = df['cash_in_bank_loan'] + df['cash_in_internal_loan']
        df['CM2_bank_loan'] = df['cash_in_bank_loan'] * (1 - df['sm'] - df['cogs'] - df['ops'] - df['bank_loan_commission'] - df['ya_split_comission'])
        df['CM2_internal_loan'] = df['cash_in_internal'] * (1 - df['sm'] - df['cogs'] - df['ops'] - df['internal_loan_commission'])
        df['CM2'] = df['CM2_bank_loan'] + df['CM2_internal_loan']
    else:
        df['cash_in_bank_loan'] = df['bank_payments'] * df['ATV_bank_loan'] * (1 + df['bank_loan_commission'])
        df['cash_in_internal_loan'] = df['installment_payments'] * df['FTV_internal_loan'] * df['internal_loan_repayment_rate'] * df['inflation_adjustment_factor']
        df['GMV'] = df['cash_in_bank_loan'] + df['cash_in_internal_loan']
        df['CM2_bank_loan'] = df['cash_in_bank_loan'] * (1 - df['sm'] - df['cogs'] - df['ops'] - df['bank_loan_commission'] - df['ya_split_comission'])
        df['CM2_internal_loan'] = df['cash_in_internal_loan'] * (1 - df['sm'] - df['cogs'] - df['ops'] - df['internal_loan_commission'])
        df['CM2'] = df['CM2_bank_loan'] + df['CM2_internal_loan']
    
    df['bank_loan_share_of_cm2'] = round(df['CM2_bank_loan'] * 100 / df['CM2'], 2)
    df['internal_loan_share_of_cm2'] = round(df['CM2_internal_loan'] * 100 / df['CM2'], 2)
    
    dff = df[['portfolio','CM2_bank_loan','CM2_internal_loan','CM2','bank_loan_share_of_cm2','internal_loan_share_of_cm2']]
    return dff

# Функция расчета GMV без компонента оплаты картой
def calculate_GMV_no_card(df, weights, internal_loan_repayment_rate, include_refunds):
    weight_bank_loan, weight_internal_loan = weights
    
    C1 = df['conv'] * (weight_bank_loan * df['CR_bank_loan'] + weight_internal_loan * df['CR_internal_loan'])
    orders_bank_loan = df['orders'] * C1 * weight_bank_loan
    orders_internal_loan = df['orders'] * C1 * weight_internal_loan
    
    if include_refunds:
        adjusted_cash_in_bank_loan = orders_bank_loan * df['ATV_bank_loan'] * (1 + df['bank_loan_commission']) * (1 - df['bank_loan_refunds_share'])
        adjusted_cash_in_internal_loan = orders_internal_loan * df['FTV_internal_loan'] * internal_loan_repayment_rate * df['inflation_adjustment_factor'] * (1 - df['internal_loan_refunds_share'])
    else:
        adjusted_cash_in_bank_loan = orders_bank_loan * df['ATV_bank_loan'] * (1 + df['bank_loan_commission'])
        adjusted_cash_in_internal_loan = orders_internal_loan * df['FTV_internal_loan'] * internal_loan_repayment_rate * df['inflation_adjustment_factor']
    
    GMV = adjusted_cash_in_bank_loan + adjusted_cash_in_internal_loan
    
    return GMV, adjusted_cash_in_bank_loan, adjusted_cash_in_internal_loan

# Функция расчета CM2 без компонента оплаты картой
def calculate_CM2(df, weights, internal_loan_repayment_rate, include_refunds):
    weight_bank_loan, weight_internal_loan = weights
    
    C1 = df['conv'] * (weight_bank_loan * df['CR_bank_loan'] + weight_internal_loan * df['CR_internal_loan'])
    orders_bank_loan = df['orders'] * C1 * weight_bank_loan
    orders_internal_loan = df['orders'] * C1 * weight_internal_loan
    
    if include_refunds:
        adjusted_cash_in_bank_loan = orders_bank_loan * df['ATV_bank_loan'] * (1 + df['bank_loan_commission']) * (1 - df['bank_loan_refunds_share'])
        adjusted_cash_in_internal_loan = orders_internal_loan * df['FTV_internal_loan'] * internal_loan_repayment_rate * df['inflation_adjustment_factor'] * (1 - df['internal_loan_refunds_share'])
        adjusted_cm2_bank_loan = adjusted_cash_in_bank_loan * (1 - df['sm'] - df['cogs'] - df['ops'] - df['bank_loan_commission'] - df['ya_split_comission'])
        adjusted_cm2_internal_loan = adjusted_cash_in_internal_loan * (1 - df['sm'] - df['cogs'] - df['ops'] - df['internal_loan_commission'])
    else:
        adjusted_cash_in_bank_loan = orders_bank_loan * df['ATV_bank_loan'] * (1 + df['bank_loan_commission'])
        adjusted_cash_in_internal_loan = orders_internal_loan * df['FTV_internal_loan'] * internal_loan_repayment_rate * df['inflation_adjustment_factor']
        adjusted_cm2_bank_loan = adjusted_cash_in_bank_loan * (1 - df['sm'] - df['cogs'] - df['ops'] - df['bank_loan_commission'] - df['ya_split_comission'])
        adjusted_cm2_internal_loan = adjusted_cash_in_internal_loan * (1 - df['sm'] - df['cogs'] - df['ops'] - df['internal_loan_commission'])   

    # Расчет CM2
    CM2 = adjusted_cm2_bank_loan + adjusted_cm2_internal_loan
    
    return CM2, adjusted_cm2_bank_loan, adjusted_cm2_internal_loan

# Основная функция приложения
def main():
    st.title("Анализ влияния рассрочки на маржинальность (CM2)")

    # Справочная информация
    st.sidebar.title("Справочная информация")
    st.sidebar.write("### Ссылки")
    st.sidebar.write("- [Детализированная воронка С1](https://datalens.yandex.cloud/pjn3nzt043p8e-detalizirovannaya-voronka-acquisition-s1)")
    st.sidebar.write("- [Влияние рассрочки на маржинальность](https://datalens.yandex.cloud/hh1fdcdlxcsg4-vliyanie-rassrochki-na-marzhinalnost)")
    st.sidebar.write("- [Воронка банковской рассрочки](https://datalens.yandex.cloud/bbs9wgyas0ns0-voronka-bankovskoy-rassrochki)")
    st.sidebar.write("- [Acquisition Funnel B2C](https://datalens.yandex.cloud/8adnr8ow39kux-acquisition-funnel-b2c)")
    st.sidebar.write("- [Consolidated 2024-25](https://docs.google.com/spreadsheets/d/1Q9RP1wJOTvEzOHFshKEWOzWJsJpPTNI2f2uIULBNvtU/edit?gid=1753026548#gid=1753026548)")
    st.sidebar.write("- [Consolidated 23-24](https://docs.google.com/spreadsheets/d/1g3mAswh1iweDZHsmRMBM5NjkT2va7ITUmFv_wXM8hfY/edit?gid=245601635#gid=245601635)")

    st.sidebar.write("### Названия полей")
    st.sidebar.write("'portfolio' - портфель")
    st.sidebar.write("'orders' - заявки")
    st.sidebar.write("'conv' - конверсия в оплату")
    st.sidebar.write("'approve_rate' - доля оппрува банковской рассрочки")
    st.sidebar.write("'take_rate' - доля взятых банковских рассрочек")
    st.sidebar.write("'CR_bank_loan' - конверсия для банковской рассрочки")
    st.sidebar.write("'CR_internal_loan' - конверсия для внутренней рассрочки")
    st.sidebar.write("'ATV_bank_loan' - средний чек для банковской рассрочки")
    st.sidebar.write("'ATV_internal_loan' - средний чек для внутренней рассрочки")
    st.sidebar.write("'bank_loan_commission' - комиссия банка за рассрочку")
    st.sidebar.write("'internal_loan_commission' - комиссия за эквайринг для внутренней рассрочки")
    st.sidebar.write("'FTV_internal_loan' - полная сумма внутренней рассрочки")
    st.sidebar.write("'annual_inflation_rate' - годовая инфляция")
    st.sidebar.write("'average_loan_period_months' - средний период рассрочки")
    st.sidebar.write("'sm' - доля затрат на продажи и маркетинг")
    st.sidebar.write("'cogs' - доля затрат на учителей")
    st.sidebar.write("'ops' - доля операционных затрат")
    st.sidebar.write("'ya_split_comission' - комиссия за Яндекс Сплит")
    st.sidebar.write("'bank_loan_refunds_share' - доля возвратов для банковской рассрочки")
    st.sidebar.write("'internal_loan_refunds_share' - доля возвратов для внутренней рассрочки")
    st.sidebar.write("'bank_payments' - количество платежей по банковской рассрочке")
    st.sidebar.write("'installment_payments' - количество платежей по внутренней рассрочке")
    st.sidebar.write("'internal_loan_repayment_rate' - фактическая выплачиваемость внутренней рассрочки")
    st.sidebar.write("'months_inflation_rate' - месячная инфляция")
    st.sidebar.write("'inflation_adjustment_factor' - поправка на инфляцию")
    
    # Фильтр по портфелю
    selected_portfolio = st.selectbox("Выберите портфель", df['portfolio'].unique())
    filtered_df = df[df['portfolio'] == selected_portfolio].copy()
    
    include_refunds = st.checkbox("Включить возвраты в расчет", value=False)
    
    # Диапазон для коэффициента выплачиваемости внутренней рассрочки
    start_range = st.slider("Начало диапазона коэффициента выплачиваемости внутренней рассрочки", 0.0, 1.0, 0.50)
    end_range = st.slider("Конец диапазона коэффициента выплачиваемости внутренней рассрочки", 0.50, 1.0, 1.0)
    
    # Создание диапазона значений internal_loan_repayment_rate
    internal_loan_repayment_rate_range_50_100 = np.arange(start_range, end_range, 0.01)
    
    # Перерасчет таблицы для каждого значения internal_loan_repayment_rate в заданном диапазоне
    results_no_card_50_100 = []

    weight_range = np.linspace(0, 1, 100)

    for repayment_rate in internal_loan_repayment_rate_range_50_100:
        best_GMV = -np.inf
        best_weights = None
        best_components = None
        
        for w1 in weight_range:
            w2 = 1 - w1  # Вся оставшаяся доля идет на внутреннюю рассрочку
            GMV, bank_loan_component, internal_loan_component = calculate_GMV_no_card(
                filtered_df.iloc[0],
                [w1, w2],
                repayment_rate,
                include_refunds=include_refunds
            )
            if GMV > best_GMV:
                best_GMV = GMV
                best_weights = [w1, w2]
                best_components = [bank_loan_component, internal_loan_component]
        
        if best_weights is not None:
            optimal_weight_bank_loan, optimal_weight_internal_loan = best_weights
            bank_loan_component, internal_loan_component = best_components
            results_no_card_50_100.append({
                "Internal Loan Repayment Rate": repayment_rate,
                "Optimal Bank Loan Weight": optimal_weight_bank_loan,
                "Optimal Internal Loan Weight": optimal_weight_internal_loan,
                "Maximized GMV": best_GMV,
                "Bank Loan GMV Component": bank_loan_component,
                "Internal Loan GMV Component": internal_loan_component
            })

    df_results_no_card_50_100 = pd.DataFrame(results_no_card_50_100)

    # Перерасчет таблицы для каждого значения internal_loan_repayment_rate для CM2
    results_cm2_50_100 = []

    for repayment_rate in internal_loan_repayment_rate_range_50_100:
        best_CM2 = -np.inf
        best_weights = None
        best_components = None
        
        for w1 in weight_range:
            w2 = 1 - w1  # Вся оставшаяся доля идет на внутреннюю рассрочку
            CM2, bank_loan_component, internal_loan_component = calculate_CM2(
                filtered_df.iloc[0],
                [w1, w2],
                repayment_rate,
                include_refunds=include_refunds
            )
            if CM2 > best_CM2:
                best_CM2 = CM2
                best_weights = [w1, w2]
                best_components = [bank_loan_component, internal_loan_component]
        
        if best_weights is not None:
            optimal_weight_bank_loan, optimal_weight_internal_loan = best_weights
            bank_loan_component, internal_loan_component = best_components
            results_cm2_50_100.append({
                "Internal Loan Repayment Rate": repayment_rate,
                "Optimal Bank Loan Weight": optimal_weight_bank_loan,
                "Optimal Internal Loan Weight": optimal_weight_internal_loan,
                "Maximized CM2": best_CM2,
                "Bank Loan CM2 Component": bank_loan_component,
                "Internal Loan CM2 Component": internal_loan_component
            })

    df_results_cm2_50_100 = pd.DataFrame(results_cm2_50_100)
    
    # Отображение таблицы с основными данными
    st.subheader("Основные данные по портфелям")
    st.dataframe(df)

    # Отображение таблицы с расчетами доли БР и ВР от CM2
    st.subheader("Фактическая доля БР и ВР от CM2 по портфелям")
    st.dataframe(calculate_financial_metrics(df, include_refunds))

    # График GMV и его компоненты
    st.subheader("GMV и его компоненты в зависимости от коэффициента выплачиваемости внутренней рассрочки")
    
    # Данные для графика
    repayment_rates = df_results_no_card_50_100["Internal Loan Repayment Rate"]
    gmv_values = df_results_no_card_50_100["Maximized GMV"]
    bank_components = df_results_no_card_50_100["Bank Loan GMV Component"]
    internal_components = df_results_no_card_50_100["Internal Loan GMV Component"]

    # Построение графиков
    plt.figure(figsize=(14, 8))

    # График общего GMV
    plt.plot(repayment_rates, gmv_values, label="Maximized GMV", marker='o')
    plt.plot(repayment_rates, bank_components, label="Bank Loan GMV Component", marker='x')
    plt.plot(repayment_rates, internal_components, label="Internal Loan GMV Component", marker='s')

    # Добавление деталей графика
    plt.xlabel("Internal Loan Repayment Rate")
    plt.ylabel("GMV (₽)")
    plt.title("GMV и его компоненты в зависимости от коэффициента выплачиваемости внутренней рассрочки")
    plt.legend()
    plt.grid(True)
    
    # Отображение графика в Streamlit
    st.pyplot(plt)

    # Отображение таблицы результатов анализа GMV
    st.subheader("Результаты анализа GMV")
    st.dataframe(df_results_no_card_50_100)

    # График CM2 и его компоненты
    st.subheader("CM2 и его компоненты в зависимости от коэффициента выплачиваемости внутренней рассрочки")

    # Данные для графика
    repayment_rates = df_results_cm2_50_100["Internal Loan Repayment Rate"]
    cm2_values = df_results_cm2_50_100["Maximized CM2"]
    bank_components = df_results_cm2_50_100["Bank Loan CM2 Component"]
    internal_components = df_results_cm2_50_100["Internal Loan CM2 Component"]

    # Построение графиков
    plt.figure(figsize=(14, 8))

    # График общего CM2
    plt.plot(repayment_rates, cm2_values, label="Maximized CM2", marker='o')
    plt.plot(repayment_rates, bank_components, label="Bank Loan CM2 Component", marker='x')
    plt.plot(repayment_rates, internal_components, label="Internal Loan CM2 Component", marker='s')

    # Добавление деталей графика
    plt.xlabel("Internal Loan Repayment Rate")
    plt.ylabel("CM2 (₽)")
    plt.title("CM2 и его компоненты в зависимости от коэффициента выплачиваемости внутренней рассрочки")
    plt.legend()
    plt.grid(True)
    
    # Отображение графика в Streamlit
    st.pyplot(plt)

    # Отображение таблицы результатов анализа CM2
    st.subheader("Результаты анализа CM2")
    st.dataframe(df_results_cm2_50_100)

if __name__ == "__main__":
    main()
