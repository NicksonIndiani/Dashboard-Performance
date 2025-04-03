#----------------------------------------------------------------------------------------------------------------------------------------------
# Importação de bibliotecas
#----------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain


#----------------------------------------------------------------------------------------------------------------------------------------------
# Configuração da página
#----------------------------------------------------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard de ROAS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

colored_header(
    label="Dashboard de ROAS por Coorte",
    description="Análise interativa da performance real e preditiva das campanhas, com simulações, projeções e detecção de outliers.",
    color_name="blue-70"
)
# Efeito inicial sutil (pode comentar depois de impressionar 😄)
rain(emoji="💡", font_size=18, falling_speed=5, animation_length="short")


with st.sidebar:
    st.image("image/avatar-01.svg", width=100)  # Ajuste a largura conforme necessário


    
    
with st.sidebar:    
    with st.container(border=True):
        st.markdown("##### 🟦 Envie o arquivo ROAS")
        roas_file = st.file_uploader("📤 Arraste ou selecione o arquivo ROAS (.csv)", type=["csv"], key="roas_upload")

    with st.container(border=True):
        st.markdown("##### 🟨 Envie o arquivo Preditivo")
        preditivo_file = st.file_uploader("📤 Arraste ou selecione o arquivo Preditivo (.csv)", type=["csv"], key="preditivo_upload")

if roas_file and preditivo_file:
    df_roas = pd.read_csv(roas_file)
    df_preditivo = pd.read_csv(preditivo_file)

    df_roas['Origem'] = 'ROAS'
    df_preditivo['Origem'] = 'Preditivo'
    df_unificado = pd.concat([df_roas, df_preditivo], ignore_index=True)

    def preparar_dataframe(df):
        semana_cols = [col for col in df.columns if 'Semana' in col]
        df.columns = [col.replace('Semana ', 'S-') if 'Semana' in col else col for col in df.columns]
        semana_cols = [col.replace('Semana ', 'S-') for col in semana_cols]
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

    with st.sidebar:
        st.markdown("## 📅 Filtro de Data")
        data_min = df_unificado['Data Início'].min()
        data_max = df_unificado['Data Início'].max()
        data_range = st.date_input("Selecione o período:", [data_min, data_max], format="DD/MM/YYYY")

    if len(data_range) == 2:
        df_unificado = df_unificado[(df_unificado['Data Início'] >= pd.to_datetime(data_range[0])) &
                                     (df_unificado['Data Início'] <= pd.to_datetime(data_range[1]))]

    with st.sidebar:
        st.markdown("## 🎯 Simulação Ao Vivo")
        semanas_restantes = st.slider("Semanas Restantes até 11", min_value=1, max_value=11, value=6)
        fase_simulada = st.selectbox("Fase da simulação", ['Alta', 'Queda'])

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

    with st.sidebar:
        st.metric("ROAS Mínimo Ideal Hoje", f"{roas_minimo_hoje:.2f}")






#----------------------------------------------------------------------------------------------------------------------------------------------
# Crescimento percentual semanal (média) para ROAS
#----------------------------------------------------------------------------------------------------------------------------------------------    
    
    
    st.header("Crescimento percentual semanal do ROAS")

    # Calcular o crescimento percentual
    crescimento_roas_absoluto = df_roas[semana_cols].mean()
    crescimento_roas_pct = crescimento_roas_absoluto.pct_change().fillna(0) * 100  # Converter para %

    df_crescimento_pct = pd.DataFrame({
        'Semana': semana_cols,
        'Crescimento (%)': crescimento_roas_pct.values
    })

    # Gráfico com Altair
    chart = alt.Chart(df_crescimento_pct).mark_line(point=alt.OverlayMarkDef(filled=True, size=60)).encode(
        x=alt.X('Semana:N', title='Semana'),
        y=alt.Y('Crescimento (%):Q', title='Crescimento Percentual'),
        tooltip=['Semana', 'Crescimento (%)']
    ).properties(
        width=700,
        height=300,
    )
    
    st.altair_chart(chart, use_container_width=True)



#----------------------------------------------------------------------------------------------------------------------------------------------
# Crescimento médio por Semana (Preditivo x ROAS)
#----------------------------------------------------------------------------------------------------------------------------------------------   
    
    
    st.header("Crescimento médio por Semana (Preditivo x ROAS)")

    medias_semanais = df_unificado.groupby('Origem')[semana_cols].mean().T
    medias_semanais.index.name = 'Semana'
    df_medias_long = medias_semanais.reset_index().melt(id_vars='Semana', var_name='Origem', value_name='Média')

    chart = alt.Chart(df_medias_long).mark_line(point=alt.OverlayMarkDef(filled=True, size=60)).encode(
        x=alt.X('Semana:N', title='Semana'),
        y=alt.Y('Média:Q', title='Média dos valores', scale=alt.Scale(zero=False)),
        color=alt.Color('Origem:N', title='Origem'),
        tooltip=['Semana', 'Origem', 'Média']
    ).properties(
        width=800,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

    
    
#----------------------------------------------------------------------------------------------------------------------------------------------
# Simulação de curva ajustada para alcançar ROAS 1.34
#----------------------------------------------------------------------------------------------------------------------------------------------    
    
    
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

    simulacao_0481 = [0.481]
    for taxa in crescimento_percentual[1:]:
        simulacao_0481.append(simulacao_0481[-1] * (1 + taxa))

    simulacao_ajustada = [valor_ideal_exato]
    for taxa in crescimento_percentual[1:]:
        simulacao_ajustada.append(simulacao_ajustada[-1] * (1 + taxa))

    df_plot = pd.DataFrame({
        'Semana': semana_cols,
        'Crescimento Real Médio': crescimento_roas_absoluto.values,
        'Simulação com valor inicial ideal': simulacao_0481,
        f'Simulação Ideal Ajustada (~{valor_ideal_exato})': simulacao_ajustada
    })

    df_long = df_plot.melt(id_vars='Semana', var_name='Tipo', value_name='ROAS')

    df_meta = pd.DataFrame({
        'Semana': semana_cols,
        'Tipo': 'Meta Final (1,34)',
        'ROAS': [1.34] * len(semana_cols)
    })

    df_final = pd.concat([df_long, df_meta], ignore_index=True)

    stroke_dash_scale = alt.Scale(domain=['Meta Final (1,34)'], range=[[4, 4]])

    chart = alt.Chart(df_final).mark_line(point=alt.OverlayMarkDef(filled=True, size=60)).encode(
        x=alt.X('Semana:N', title='Semana', sort=semana_cols),
        y=alt.Y('ROAS:Q', title='ROAS'),
        color=alt.Color('Tipo:N', title='Simulação'),
        strokeDash=alt.StrokeDash('Tipo:N', scale=stroke_dash_scale),
        tooltip=['Semana', 'Tipo', alt.Tooltip('ROAS:Q', format='.2f')]
    ).properties(
        width=900,
        height=400
    )

    col1, _ = st.columns(2)
    with col1:
        st.metric(label="🎯 Valor Inicial Ideal", value=f"{valor_ideal_exato:.2f}", delta="Meta final: 1,34")

    st.altair_chart(chart, use_container_width=True)

    
    
#----------------------------------------------------------------------------------------------------------------------------------------------
# ROI por faixa de investimento
#----------------------------------------------------------------------------------------------------------------------------------------------    
    
    
    st.header("ROAS médio por faixa de investimento")

    bins = [0, 100000, 150000, 200000, 250000, 300000, 400000]
    labels = ["até R$100k", "R$100k–150k", "R$150k–200k", "R$200k–250k", "R$250k–300k", "acima de R$300k"]
    df_unificado["Faixa de Investimento"] = pd.cut(df_unificado["Custo"], bins=bins, labels=labels)

    roas_por_faixa = df_unificado.groupby("Faixa de Investimento")["ROAS"].agg(['mean', 'count']).reset_index()
    roas_por_faixa.columns = ['Faixa', 'ROAS Médio', 'Quantidade']

    bar = alt.Chart(roas_por_faixa).mark_bar().encode(
        x=alt.X('Faixa:N', title='Faixa de Investimento', sort=labels),
        y=alt.Y('ROAS Médio:Q', title='ROAS Médio'),
        tooltip=['Faixa', 'ROAS Médio', 'Quantidade'],
        color=alt.value("#4F81BD")  
    )

    linha_meta = alt.Chart(pd.DataFrame({'y': [1.34]})).mark_rule(
        color='red', strokeDash=[4, 4]
    ).encode(y='y:Q')

    texto_meta = alt.Chart(pd.DataFrame({'y': [1.34]})).mark_text(
        text='Meta ROAS (1.34)', align='left', dx=5, dy=-5, color='red'
    ).encode(
        y='y:Q',
        x=alt.value(0)
    )

    grafico = (bar + linha_meta + texto_meta).properties(
        width=800,
        height=400,
    ).configure_axisX(labelAngle=0)

    st.altair_chart(grafico, use_container_width=True)




#----------------------------------------------------------------------------------------------------------------------------------------------
# ROAS médio por fase
#----------------------------------------------------------------------------------------------------------------------------------------------   
    
    
    st.header("📊 ROAS Médio por Fase Sazonal")

    media_roas_fase = df_unificado.groupby("Fase Semestral")["ROAS"].mean().reset_index()

    bar = alt.Chart(media_roas_fase).mark_bar().encode(
        x=alt.X('Fase Semestral:N', title='Fase Sazonal', sort=['Baixa', 'Alta', 'Queda']),
        y=alt.Y('ROAS:Q', title='ROAS Médio'),
        tooltip=['Fase Semestral', alt.Tooltip('ROAS:Q', format='.2f')],
        color=alt.value("#4F81BD")
    )

    linha_meta = alt.Chart(pd.DataFrame({'y': [1.34]})).mark_rule(
        color='red', strokeDash=[4, 4]
    ).encode(y='y:Q')

    texto_meta = alt.Chart(pd.DataFrame({'y': [1.34]})).mark_text(
        text='Meta ROAS (1.34)', align='left', dx=5, dy=-5, color='red'
    ).encode(
        y='y:Q',
        x=alt.value(0)
    )

    grafico_fase = (bar + linha_meta + texto_meta).properties(
        width=700,
        height=400,
        title="ROAS Médio por Fase"
    )

    st.altair_chart(grafico_fase, use_container_width=True)

    
    
    
#----------------------------------------------------------------------------------------------------------------------------------------------
# Semanas até atingir ROAS 1.34
#----------------------------------------------------------------------------------------------------------------------------------------------   
    
    
    st.header("⏳ Semanas até atingir ROAS 1.34")

    def semana_atinge_134(linha):
        for i, val in enumerate(linha[semana_cols]):
            if val >= 1.34:
                return i + 1
        return np.nan

    df_roas['Semana Atinge 1.34'] = df_roas.apply(semana_atinge_134, axis=1)
    tempo_medio = df_roas['Semana Atinge 1.34'].mean()
    pct_nao_atingiram = df_roas['Semana Atinge 1.34'].isna().mean() * 100

    col1, col2 = st.columns(2)
    col1.metric("📈 Semana Média para Atingir 1.34", round(tempo_medio, 2))
    col2.metric("❌ % de Coortes que NÃO Atingem", f"{pct_nao_atingiram:.1f}%")

    # Distribuição das semanas até atingir
    df_dist = df_roas['Semana Atinge 1.34'].fillna('Não Atingiu')
    df_dist = df_dist.value_counts().reset_index()
    df_dist.columns = ['Semana', 'Coortes']

    df_dist['Semana'] = df_dist['Semana'].astype(str)

    chart = alt.Chart(df_dist).mark_bar().encode(
        x=alt.X('Semana:N', title='Semana em que Atingiu 1.34'),
        y=alt.Y('Coortes:Q', title='Quantidade de Coortes'),
        color=alt.condition(
            alt.datum['Semana'] == 'Não Atingiu',
            alt.value('crimson'),
            alt.value('#4F81BD')
        ),
        tooltip=['Semana', 'Coortes']
    ).properties(
        width=750,
        height=400,
        title="Distribuição das Semanas até ROAS 1.34"
    )

    st.altair_chart(chart, use_container_width=True)




#----------------------------------------------------------------------------------------------------------------------------------------------
# Projeção de ROAS final com base no crescimento real da coorte
#----------------------------------------------------------------------------------------------------------------------------------------------    
    
    
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

    pct_proj_acima_meta = (df_roas['ROAS Projetado'] >= 1.34).mean() * 100
    media_roas_proj = df_roas['ROAS Projetado'].mean()

    col1, col2 = st.columns(2)
    col1.metric("📈 ROAS Projetado Médio", f"{media_roas_proj:.2f}")
    col2.metric("✅ % de Projeções acima da Meta", f"{pct_proj_acima_meta:.1f}%")

    # Faixas personalizadas
    bins = [0.4, 1.0, 1.34, 1.7, 2.0, 2.5, 3.0]
    labels = ['0.4 – 1.0', '1.0 – 1.34', '1.34 – 1.7', '1.7 – 2.0', '2.0 – 2.5', '2.5 – 3.0']
    df_roas['Faixa ROAS Projetado'] = pd.cut(df_roas['ROAS Projetado'], bins=bins, labels=labels, include_lowest=True)

    df_hist = df_roas['Faixa ROAS Projetado'].value_counts().sort_index().reset_index()
    df_hist.columns = ['Faixa', 'Coortes']

    chart = alt.Chart(df_hist).mark_bar().encode(
        x=alt.X('Faixa:N', title='Faixa de ROAS Projetado'),
        y=alt.Y('Coortes:Q', title='Quantidade de Coortes'),
        tooltip=['Faixa', 'Coortes'],
        color=alt.value("#4F81BD")
    ).properties(
        width=800,
        height=400,
        title="Distribuição do ROAS Projetado por Coorte"
    )

    st.altair_chart(chart, use_container_width=True)




#---------------------------------------------------------------------------------------------------------------------------------------------
# Outliers - Crescimentos extremos
#---------------------------------------------------------------------------------------------------------------------------------------------


    st.header("🔎 Análise de Outliers - Crescimento (%)")

    df_unificado['Crescimento (%)'] = df_unificado.apply(
        lambda row: (row[semana_cols[-1]] - row[semana_cols[0]]) / row[semana_cols[0]]
        if pd.notnull(row[semana_cols[0]]) and pd.notnull(row[semana_cols[-1]])
        else np.nan,
        axis=1
    )

    q1 = df_unificado['Crescimento (%)'].quantile(0.25)
    q3 = df_unificado['Crescimento (%)'].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    df_unificado['É Outlier'] = df_unificado['Crescimento (%)'].apply(
        lambda x: 'Outlier' if x < limite_inferior or x > limite_superior else 'Normal'
    )

    # Gráfico com Altair
    hist = alt.Chart(df_unificado.dropna(subset=['Crescimento (%)'])).mark_bar().encode(
        x=alt.X('Crescimento (%):Q', bin=alt.Bin(maxbins=30), title='Crescimento (%)'),
        y=alt.Y('count():Q', title='Quantidade de Coortes'),
        color=alt.Color('É Outlier:N', scale=alt.Scale(domain=['Outlier', 'Normal'], range=['crimson', '#4F81BD'])),
        tooltip=['count()']
    ).properties(
        width=800,
        height=400,
        title="Distribuição do Crescimento (%) com Destaque para Outliers"
    )

    st.altair_chart(hist, use_container_width=True)

    # Mostrar tabela somente se houver outliers
    outliers = df_unificado[df_unificado['É Outlier'] == 'Outlier']

    if not outliers.empty:
        st.subheader("Coortes com crescimento fora do padrão:")
        st.dataframe(outliers[['Periodo ( 7 dias)', 'Crescimento (%)', 'Origem', 'Custo', 'Receita']])
    else:
        st.info("Nenhum outlier encontrado com base no IQR.")

