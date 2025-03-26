import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard de ROAS", layout="wide")

st.title("Análise de ROAS por Coorte")

st.sidebar.header("📁 Upload dos Arquivos")
roas_file = st.sidebar.file_uploader("Envie o arquivo ROAS", type=["csv"])
preditivo_file = st.sidebar.file_uploader("Envie o arquivo Preditivo", type=["csv"])

if roas_file and preditivo_file:
    df_roas = pd.read_csv(roas_file)
    df_preditivo = pd.read_csv(preditivo_file)

    df_roas['Origem'] = 'ROAS'
    df_preditivo['Origem'] = 'Preditivo'
    df_unificado = pd.concat([df_roas, df_preditivo], ignore_index=True)

    def preparar_dataframe(df):
        semana_cols = [col for col in df.columns if 'Semana' in col]
        df.columns = [col.replace('Semana ', 'S') if 'Semana' in col else col for col in df.columns]
        semana_cols = [col.replace('Semana ', 'S') for col in semana_cols]
        df[semana_cols] = df[semana_cols].replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce')
        df['Custo'] = df['Custo'].replace({r'R\$': '', r'\.': '', ',': '.'}, regex=True).astype(float)
        df['Receita'] = df['Receita'].replace({r'R\$': '', r'\.': '', ',': '.'}, regex=True).astype(float)

        df['Data Início'] = pd.to_datetime(df['Periodo ( 7 dias)'].str.extract(r'(\d{2}/\d{2}/\d{4})')[0], dayfirst=True)
        def classificar_fase(data):
            mes = data.month
            if mes in [10, 11, 12]: return 'Baixa'
            elif mes in [1, 2]: return 'Alta'
            elif mes == 3: return 'Queda'
            return 'Outro'
        df['Fase Semestral'] = df['Data Início'].apply(classificar_fase)
        return df, semana_cols

    df_unificado, semana_cols = preparar_dataframe(df_unificado)
    df_roas, _ = preparar_dataframe(df_roas)
    df_preditivo, _ = preparar_dataframe(df_preditivo)

    df_unificado['ROAS'] = df_unificado['Receita'] / df_unificado['Custo']

    st.sidebar.header("📅 Filtro de Data")
    data_min = df_unificado['Data Início'].min()
    data_max = df_unificado['Data Início'].max()
    data_range = st.sidebar.date_input("Selecione o período:", [data_min, data_max], format="DD/MM/YYYY")

    if len(data_range) == 2:
        df_unificado = df_unificado[(df_unificado['Data Início'] >= pd.to_datetime(data_range[0])) &
                                     (df_unificado['Data Início'] <= pd.to_datetime(data_range[1]))]

    st.sidebar.header("🎯 Simulação Ao Vivo")
    semanas_restantes = st.sidebar.slider("Semanas Restantes até 11", min_value=1, max_value=11, value=6)
    fase_simulada = st.sidebar.selectbox("Fase da simulação", ['Alta', 'Queda'])

    crescimento_fase = {
        "Alta": [0.38, 0.15, 0.11, 0.095, 0.065, 0.045, 0.025, 0.02, 0.05, 0.0],
        "Queda": [0.38, -0.07, 0.0, 0.0, 0.01, 0.01, 0.005, 0.005, 0.0, 0.0]
    }
    crescimento_selecionado = crescimento_fase[fase_simulada][-semanas_restantes:]

    def calcular_roas_minimo_para_meta(meta_final, crescimentos):
        valor = meta_final
        for taxa in reversed(crescimentos):
            valor /= (1 + taxa)
        return round(valor, 4)

    roas_minimo_hoje = calcular_roas_minimo_para_meta(1.34, crescimento_selecionado)
    st.sidebar.metric("ROAS Mínimo Ideal Hoje", f"{roas_minimo_hoje:.2f}")

    # Crescimento percentual semanal (média) para ROAS
    st.header("Crescimento percentual semanal do ROAS")
    crescimento_roas_absoluto = df_roas[semana_cols].mean()
    crescimento_roas_pct = crescimento_roas_absoluto.pct_change().fillna(0)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=semana_cols, y=crescimento_roas_pct.values, marker='o', ax=ax3)
    ax3.set_title("Crescimento Percentual Semanal do ROAS")
    ax3.set_xlabel("Semana")
    ax3.set_ylabel("Crescimento %")
    ax3.grid(True)
    st.pyplot(fig3)

    # Crescimento médio por Semana (Preditivo x ROAS)
    st.header("Crescimento médio por Semana (Preditivo x ROAS)")
    medias_semanais = df_unificado.groupby('Origem')[semana_cols].mean().T

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    medias_semanais.plot(marker='o', ax=ax4)
    ax4.set_title("Crescimento médio por Semana (Preditivo x ROAS)")
    ax4.set_ylabel("Média dos valores")
    ax4.set_xlabel("Semana")
    ax4.grid(True)
    st.pyplot(fig4)

    # Simulação de curva ajustada para alcançar ROAS 1.34
    st.header("Simulação Ajustada para Alcançar ROAS 1,34")
    crescimento_percentual = crescimento_roas_absoluto.pct_change().fillna(0).values

    def encontrar_valor_inicial_exato(crescimento, alvo=1.34, precisao=0.0001):
        baixo, alto = 0.4, 1.0
        while alto - baixo > precisao:
            meio = (baixo + alto) / 2
            valor = meio
            for taxa in crescimento[1:]:
                valor *= (1 + taxa)
            if valor >= alvo:
                alto = meio
            else:
                baixo = meio
        return round((baixo + alto) / 2, 4)

    valor_ideal_exato = encontrar_valor_inicial_exato(crescimento_percentual)
    simulado_exato = [valor_ideal_exato]
    for taxa in crescimento_percentual[1:]:
        simulado_exato.append(simulado_exato[-1] * (1 + taxa))

    df_plot = pd.DataFrame({
        'Semana': semana_cols,
        'Crescimento Real Médio': crescimento_roas_absoluto.values,
        'Simulação com valor inicial ideal': [0.481] + list(np.cumprod([1 + t for t in crescimento_percentual[1:]]) * 0.481)
    })
    df_plot['Simulação Ideal Ajustada (alvo 1,34)'] = simulado_exato

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(df_plot['Semana'], df_plot['Crescimento Real Médio'], marker='o', label='Crescimento Real Médio')
    ax5.plot(df_plot['Semana'], df_plot['Simulação com valor inicial ideal'], marker='o', linestyle='--', label='Simulação Inicial (0,481)')
    ax5.plot(df_plot['Semana'], df_plot['Simulação Ideal Ajustada (alvo 1,34)'], marker='o', linestyle='--', label=f'Simulação Ajustada (~{valor_ideal_exato})')
    ax5.axhline(1.34, color='gray', linestyle=':', label='Meta Final (1,34)')
    ax5.set_title('Simulação Ajustada para Alcançar ROAS 1,34')
    ax5.set_ylabel('ROAS')
    ax5.set_xlabel('Semana')
    ax5.set_xticklabels(df_plot['Semana'], rotation=45)
    ax5.grid(True)
    ax5.legend()
    st.pyplot(fig5)

    # ROI por faixa de investimento
    st.header("ROAS médio por faixa de investimento")
    bins = [0, 100000, 150000, 200000, 250000, 300000, 400000]
    labels = ["até R$100k", "R$100k–150k", "R$150k–200k", "R$200k–250k", "R$250k–300k", "acima de R$300k"]
    df_unificado["Faixa de Investimento"] = pd.cut(df_unificado["Custo"], bins=bins, labels=labels)
    roas_por_faixa = df_unificado.groupby("Faixa de Investimento")["ROAS"].agg(['mean', 'count']).reset_index()
    st.dataframe(roas_por_faixa)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=roas_por_faixa, x='Faixa de Investimento', y='mean', palette='Blues_d', ax=ax2)
    ax2.axhline(1.34, color='red', linestyle='--', label='Meta ROAS (1.34)')
    ax2.set_title('ROAS Médio por Faixa de Investimento')
    ax2.set_ylabel('ROAS Médio')
    ax2.set_xlabel('Faixa de Investimento')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # INSIGHT 1: ROAS médio por fase
    st.header("📊 ROAS Médio por Fase Sazonal")
    media_roas_fase = df_unificado.groupby("Fase Semestral")["ROAS"].mean().reset_index()
    st.dataframe(media_roas_fase)
    fig_fase, ax_fase = plt.subplots()
    sns.barplot(data=media_roas_fase, x='Fase Semestral', y='ROAS', ax=ax_fase, palette='pastel')
    ax_fase.axhline(1.34, color='red', linestyle='--', label='Meta ROAS')
    ax_fase.set_title("ROAS Médio por Fase")
    ax_fase.legend()
    st.pyplot(fig_fase)

    # INSIGHT 3: Semanas até atingir ROAS 1.34
    # Ele está dizendo:
    # “Das coortes que efetivamente atingiram ROAS ≥ 1.34, a média de tempo para atingir esse valor foi 5.86 semanas.”
    # Ou seja, não é uma previsão, é uma média histórica real para os casos bem-sucedidos.
    st.header("⏳ Semanas até atingir ROAS 1.34")
    def semana_atinge_134(linha):
        for i, val in enumerate(linha[semana_cols]):
            if val >= 1.34:
                return i + 1
        return np.nan

    df_roas['Semana Atinge 1.34'] = df_roas.apply(semana_atinge_134, axis=1)
    tempo_medio = df_roas['Semana Atinge 1.34'].mean()
    pct_nao_atingiram = df_roas['Semana Atinge 1.34'].isna().mean() * 100
    st.metric("Semana Média para Atingir 1.34", round(tempo_medio, 2))
    st.metric("% de Coortes que NÃO Atingem 1.34", f"{pct_nao_atingiram:.1f}%")

    # INSIGHT 4: Projeção de ROAS final com base no crescimento real da coorte
    st.header("🎯 Projeção de ROAS Final com Base no Crescimento da Coorte")

    def projetar_roas_com_base_na_coorte(linha):
        valores = linha[semana_cols].dropna().values.tolist()
        if not valores:
            return np.nan

        semanas_completas = len(valores)
        semanas_restantes = 11 - semanas_completas
        if semanas_restantes <= 0:
            return valores[-1]

        crescimentos_reais = []
        for i in range(1, semanas_completas):
            if valores[i - 1] > 0:
                crescimento = (valores[i] - valores[i - 1]) / valores[i - 1]
                crescimentos_reais.append(crescimento)

        if len(crescimentos_reais) < 1:
            return valores[-1]

        ultimos = crescimentos_reais[-2:]
        media_final = np.mean(ultimos)

        if media_final <= 0:
            return valores[-1]

        for _ in range(semanas_restantes):
            valores.append(valores[-1] * (1 + media_final))

        return valores[-1]

    df_roas['ROAS Projetado'] = df_roas.apply(projetar_roas_com_base_na_coorte, axis=1)
    st.dataframe(df_roas[['Periodo ( 7 dias)', 'ROAS Projetado']].head())

    # INSIGHT 5: Outliers - Crescimentos extremos
    st.header("🔎 Análise de Outliers - Crescimento (%)")
    df_unificado['Crescimento (%)'] = df_unificado.apply(lambda row: (row[semana_cols[-1]] - row[semana_cols[0]]) / row[semana_cols[0]] if pd.notnull(row[semana_cols[0]]) and pd.notnull(row[semana_cols[-1]]) else np.nan, axis=1)
    q1 = df_unificado['Crescimento (%)'].quantile(0.25)
    q3 = df_unificado['Crescimento (%)'].quantile(0.75)
    iqr = q3 - q1
    outliers = df_unificado[(df_unificado['Crescimento (%)'] < q1 - 1.5 * iqr) | (df_unificado['Crescimento (%)'] > q3 + 1.5 * iqr)]

    st.subheader("Coortes com crescimento fora do padrão:")
    st.dataframe(outliers[['Periodo ( 7 dias)', 'Crescimento (%)', 'Origem', 'Custo', 'Receita']])

else:
    st.info("⬅️ Por favor, envie os dois arquivos CSV para iniciar a análise.")
