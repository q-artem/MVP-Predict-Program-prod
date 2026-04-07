import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# НАСТРОЙКИ СТРАНИЦЫ
st.set_page_config(page_title="Глобальная аналитика", page_icon="📊", layout="wide")

st.title("📊 Глобальный анализ учебных потоков")

if 'student_df' not in st.session_state or st.session_state['student_df'] is None:
    st.warning("⚠️ Данные не найдены. Пожалуйста, вернитесь на вкладку «Загрузка данных» и загрузите файл.")
    st.stop()

df = st.session_state['student_df']

col_sem = 'НОМЕРА СЕМЕСТРА' if 'НОМЕРА СЕМЕСТРА' in df.columns else 'Семестр' if 'Семестр' in df.columns else \
df.columns[0]
col_gpa = 'Средняя_кач_пред_сем' if 'Средняя_кач_пред_сем' in df.columns else 'GPA' if 'GPA' in df.columns else \
df.columns[1]
col_debt = 'Несдано_кач_до_пред_сем' if 'Несдано_кач_до_пред_сем' in df.columns else 'Долги' if 'Долги' in df.columns else \
df.columns[2]
col_pay = 'УСЛОВИЕ_ЗАЧИСЛЕНИЯ' if 'УСЛОВИЕ_ЗАЧИСЛЕНИЯ' in df.columns else df.columns[3]

st.markdown(f"Анализ построен на базе **{len(df)}** записей.")
st.divider()

# СТРУКТУРА ГРУППЫ АКАДЕМИЧЕСКОГО РИСКА
st.subheader("Структура студентов в зоне низкой успеваемости")
st.markdown("Распределение студентов с GPA < 3.5 или имеющих исторические задолженности по условиям зачисления.")

try:
    if 'ИД' not in df.columns:
        st.error("В файле не найдена колонка 'ИД'.")
    else:
        df_unique = df.drop_duplicates(subset=['ИД']).copy()

        if 'ОЦЕНКА_NUM' in df_unique.columns:
            debt_cols = [c for c in df_unique.columns if c.startswith('Несдано_вовремя_сем_')]
            gpa_cols = [c for c in df_unique.columns if c.startswith('Средняя_оценка_сем_')]
        elif 'Оценка (0 или 1)' in df_unique.columns:
            debt_cols = [c for c in df_unique.columns if c.startswith('Несдано_бинарных_вовремя_сем_')]
            gpa_cols = [c for c in df_unique.columns if c.startswith('Средняя_оценка_кач_сем_')]
        else:
            debt_cols, gpa_cols = [], []

        if not debt_cols or not gpa_cols:
            st.warning("Не удалось распознать формат файла (не найдены колонки истории успеваемости).")
        else:
            df_unique['Total_Debts'] = df_unique[debt_cols].fillna(0).sum(axis=1)

            temp_gpa = df_unique[gpa_cols].replace(0, pd.NA)
            df_unique['Avg_GPA'] = temp_gpa.mean(axis=1).fillna(0)

            is_at_risk = (df_unique['Total_Debts'] > 0) | ((df_unique['Avg_GPA'] > 0) & (df_unique['Avg_GPA'] < 3.5))
            low_perf_df = df_unique[is_at_risk].copy()

            if low_perf_df.empty:
                st.info("Отличные новости: в текущих данных нет студентов в зоне риска!")
            else:
                paid_share = low_perf_df['УСЛОВИЕ_ЗАЧИСЛЕНИЯ'].value_counts().reset_index()
                paid_share.columns = ['УСЛОВИЕ_ЗАЧИСЛЕНИЯ', 'Количество']

                fig4 = px.pie(
                    paid_share,
                    values='Количество',
                    names='УСЛОВИЕ_ЗАЧИСЛЕНИЯ',
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )

                fig4.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    insidetextorientation='radial',
                    marker=dict(line=dict(color='#FFFFFF', width=1)),
                    textfont_size=12
                )

                fig4.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top", y=1,
                        xanchor="left", x=1.05,
                        title="Все формы зачисления:"
                    ),
                    margin=dict(t=20, b=20, l=20, r=150),
                    height=500
                )

                st.plotly_chart(fig4, use_container_width=True)

                total_risk = len(low_perf_df)
                top_category = paid_share.iloc[0]['УСЛОВИЕ_ЗАЧИСЛЕНИЯ']
                top_count = paid_share.iloc[0]['Количество']
                top_percent = (top_count / total_risk) * 100

                st.info(f"**Аналитика:** В зоне риска находится **{total_risk}** уникальных студентов. "
                        f"Наибольшую долю ({top_percent:.1f}%) составляет категория **«{top_category}»** ({top_count} чел.).")

except Exception as e:
    st.error(f"Произошла ошибка при построении графика структуры риска: {e}")

st.divider()

# РЕЙТИНГ «ПРЕДМЕТОВ-УБИЙЦ» (МАТРИЦА РИСКА ДИСЦИПЛИН)
st.subheader("Рейтинг сложности дисциплин")
st.markdown("Поиск «предметов-барьеров». Чем левее и выше находится предмет, тем больше студентов не могут его сдать.")

try:
    disc_col = 'ДИСЦИПЛИНА' if 'ДИСЦИПЛИНА' in df.columns else 'Дисциплина' if 'Дисциплина' in df.columns else None

    if 'ОЦЕНКА_NUM' in df.columns:
        score_col = 'ОЦЕНКА_NUM'
        is_binary = False
    elif 'ОЦЕНКА' in df.columns:
        score_col = 'ОЦЕНКА'
        is_binary = False
    elif 'Оценка (0 или 1)' in df.columns:
        score_col = 'Оценка (0 или 1)'
        is_binary = True
    else:
        score_col = None
        is_binary = True

    if not disc_col or not score_col:
        st.warning("В данных не найдены колонки с названиями дисциплин или оценками.")
    else:
        if not is_binary:
            subj_stats = df.groupby(disc_col).agg(
                Средний_балл=(score_col, 'mean'),
                Процент_несдавших=(score_col, lambda x: (x <= 2).mean() * 100),
                Сдавало_человек=(score_col, 'count')
            ).reset_index()
        else:
            subj_stats = df.groupby(disc_col).agg(
                Средний_балл=(score_col, 'mean'),
                Процент_несдавших=(score_col, lambda x: (x == 0).mean() * 100),
                Сдавало_человек=(score_col, 'count')
            ).reset_index()

        # Убираем курсы, где было 2-3 человека
        min_students = 10
        subj_stats = subj_stats[subj_stats['Сдавало_человек'] >= min_students]

        if subj_stats.empty:
            st.info("Недостаточно данных по дисциплинам (везде меньше 10 сдающих).")
        else:
            subj_stats['Полное_название'] = subj_stats[disc_col]
            subj_stats[disc_col] = subj_stats[disc_col].apply(
                lambda x: str(x)[:35] + '...' if len(str(x)) > 35 else str(x))

            # 5. ПОСТРОЕНИЕ ГРАФИКА
            fig1 = px.scatter(
                subj_stats,
                x='Средний_балл',
                y='Процент_несдавших',
                text=disc_col,
                size='Сдавало_человек',
                color='Процент_несдавших',
                color_continuous_scale='Reds',
                hover_name='Полное_название',
                hover_data={'Средний_балл': ':.2f', 'Процент_несдавших': ':.1f', disc_col: False},
                labels={
                    'Средний_балл': 'Средняя оценка по потоку' if not is_binary else 'Доля успешных сдач (от 0 до 1)',
                    'Процент_несдавших': 'Доля несдавших (%)',
                    'Сдавало_человек': 'Кол-во студентов'
                }
            )

            fig1.update_traces(
                textposition='top center',
                textfont=dict(size=10)
            )

            mean_fail_rate = subj_stats['Процент_несдавших'].mean()
            fig1.add_hline(
                y=mean_fail_rate,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"Средний уровень несдач ({mean_fail_rate:.1f}%)",
                annotation_position="bottom right"
            )

            fig1.update_layout(
                height=700,
                margin=dict(l=20, r=20, t=30, b=20),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig1, use_container_width=True)

            hardcore_subjects = subj_stats[subj_stats['Сдавало_человек'] > 20].nlargest(3, 'Процент_несдавших')

            analyze = "**Аналитика:** Топ-3 самых проблемных дисциплин:\n\n"

            if not hardcore_subjects.empty:
                for _, row in hardcore_subjects.iterrows():
                    analyze += f"**{row['Полное_название']}** -- не сдали {row['Процент_несдавших']:.1f}% студентов.\n\n"
            st.info(analyze)

except Exception as e:
    st.error(f"Не удалось построить рейтинг дисциплин: {e}")

st.divider()

# КОРРЕЛЯЦИОННАЯ МАТРИЦА (ВЗАИМОСВЯЗЬ ПРЕДМЕТОВ)
st.subheader("Взаимосвязь дисциплин (Корреляционная матрица)")
st.markdown(
    "Темно-красные квадраты показывают предметы, оценки по которым сильно взаимосвязаны (успешно сдал один — сдаст и другой). Синие квадраты означают обратную зависимость.")

try:
    id_col = 'ИД' if 'ИД' in df.columns else 'ИДЕНТИФИКАТОР' if 'ИДЕНТИФИКАТОР' in df.columns else df.columns[0]
    disc_col = 'ДИСЦИПЛИНА' if 'ДИСЦИПЛИНА' in df.columns else 'Дисциплина' if 'Дисциплина' in df.columns else None

    if 'ОЦЕНКА_NUM' in df.columns:
        score_col = 'ОЦЕНКА_NUM'
    elif 'ОЦЕНКА' in df.columns:
        score_col = 'ОЦЕНКА'
    elif 'Оценка (0 или 1)' in df.columns:
        score_col = 'Оценка (0 или 1)'
    else:
        score_col = None

    if not disc_col or not score_col:
        st.warning("В данных не найдены колонки с названиями дисциплин или оценками для построения матрицы.")
    else:
        with st.spinner("Рассчитываем взаимосвязи между предметами..."):
            top_50_subjects = df[disc_col].value_counts().nlargest(50).index
            df_corr = df[df[disc_col].isin(top_50_subjects)].copy()

            df_corr[disc_col] = df_corr[disc_col].apply(lambda x: str(x)[:40] + '...' if len(str(x)) > 40 else str(x))

            pivot_subj = df_corr.pivot_table(index=id_col, columns=disc_col, values=score_col)
            corr_matrix = pivot_subj.corr().round(2)

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=False,  # ВЫКЛЮЧАЕМ цифры в квадратиках (иначе будет каша)
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                labels=dict(color="Корреляция")
            )

            fig_corr.update_layout(
                height=950,
                margin=dict(l=0, r=0, b=0, t=30),
                xaxis_title=None,
                yaxis_title=None
            )

            fig_corr.update_xaxes(tickangle=45, tickfont=dict(size=10))
            fig_corr.update_yaxes(tickfont=dict(size=10))

            st.plotly_chart(fig_corr, use_container_width=True, theme="streamlit")

            # Топ 5
            corr_df = corr_matrix.copy()

            np.fill_diagonal(corr_df.values, np.nan)

            corr_df.index.name = None
            corr_df.columns.name = None

            corr_pairs = corr_df.unstack().reset_index()
            corr_pairs.columns = ['Дисциплина_1', 'Дисциплина_2', 'Корреляция']

            corr_pairs = corr_pairs.dropna()

            corr_pairs['pair_id'] = corr_pairs.apply(
                lambda row: tuple(sorted([row['Дисциплина_1'], row['Дисциплина_2']])), axis=1)
            unique_pairs = corr_pairs.drop_duplicates(subset=['pair_id']).drop(columns=['pair_id'])

            top_5_positive = unique_pairs.nlargest(5, 'Корреляция')

            if not top_5_positive.empty:

                analyze = "**Аналитика:** Топ-5 самых взаимосвязанных пар дисциплин\n\n" \
                        "Успешная сдача одного предмета из пары с высокой вероятностью гарантирует сдачу второго:\n\n"
                for idx, row in top_5_positive.iterrows():
                    analyze += f"- **{row['Дисциплина_1']}** ↔ **{row['Дисциплина_2']}** (Корреляция: {row['Корреляция']:.2f})\n\n"
                st.info(analyze)


except Exception as e:
    st.error(f"Не удалось построить корреляционную матрицу: {e}")

from plotly.subplots import make_subplots

st.divider()

# ШОКОВАЯ АДАПТАЦИЯ (ВЛИЯНИЕ ОБЩЕЖИТИЯ)
st.subheader("Влияние общежития на успеваемость")
st.markdown("Сравнение распределения среднего балла студентов, проживающих в общежитии и вне его, по каждому семестру.")

try:
    id_col = 'ИД' if 'ИД' in df.columns else 'ИДЕНТИФИКАТОР' if 'ИДЕНТИФИКАТОР' in df.columns else df.columns[0]
    dorm_col = 'ПРИЗНАК_ОБЩЕЖИТИЯ' if 'ПРИЗНАК_ОБЩЕЖИТИЯ' in df.columns else None

    gpa_cols = [c for c in df.columns if 'Средняя_оценка_сем_' in c or 'Средняя_оценка_кач_сем_' in c]

    if not dorm_col or not gpa_cols:
        st.warning("В данных не найдены колонки признака общежития или поеместровых средних оценок.")
    else:
        import re

        def extract_sem_num(col_name):
            match = re.search(r'\d+', col_name)
            return int(match.group()) if match else 0


        gpa_cols = sorted(gpa_cols, key=extract_sem_num)
        sems_count = len(gpa_cols)

        df_unique = df.drop_duplicates(subset=[id_col]).copy()

        fig6 = make_subplots(
            rows=1,
            cols=sems_count,
            subplot_titles=[f"Сем {extract_sem_num(c)}" for c in gpa_cols],
            shared_yaxes=True,
            x_title='Проживание в общежитии',
            y_title="Средняя оценка"
        )

        for q, col_name in enumerate(gpa_cols, start=1):
            df_filtered = df_unique[df_unique[col_name] > 0]

            box = px.box(
                df_filtered,
                x=dorm_col,
                y=col_name,
                color=dorm_col,
                color_discrete_sequence=['#FFA15A', '#636EFA']
            )

            for tr in box.data:
                tr.showlegend = False
                fig6.add_trace(tr, col=q, row=1)

        fig6.update_layout(
            showlegend=False,
            height=500,
            margin=dict(l=90, r=20, t=50, b=50)
        )

        fig6.update_yaxes(range=[1.9, 5.1], row=1, col=1)

        st.plotly_chart(fig6, theme="streamlit")

        st.info("**Аналитика:** В среднем, к концу обучения, студенты, проживающие вне общежития, улучшают свои оценки по сравнению со студентами, проживающими в общежитии")

except Exception as e:
    st.error(f"Не удалось построить график влияния общежития: {e}")

st.divider()

# ИНДЕКС НАГРУЗКИ
st.subheader("Индекс академической нагрузки")
st.markdown(
    "Сравнение среднего количества академических задолженностей на **одного студента** в зависимости от формы зачисления по каждому семестру.")

try:
    id_col = 'ИД' if 'ИД' in df.columns else 'ИДЕНТИФИКАТОР' if 'ИДЕНТИФИКАТОР' in df.columns else df.columns[0]
    pay_col = 'УСЛОВИЕ_ЗАЧИСЛЕНИЯ' if 'УСЛОВИЕ_ЗАЧИСЛЕНИЯ' in df.columns else 'Форма_обучения' if 'Форма_обучения' in df.columns else None

    debt_cols = [c for c in df.columns if 'Несдано_вовремя_сем_' in c or 'Несдано_бинарных_вовремя_сем_' in c]

    if not pay_col or not debt_cols:
        st.warning("Не найдены колонки с историей долгов или условиями зачисления.")
    else:
        import re

        df_unique = df.drop_duplicates(subset=[id_col]).copy()

        def extract_sem_num(col_name):
            match = re.search(r'\d+', col_name)
            return int(match.group()) if match else 0


        debt_cols = sorted(debt_cols, key=extract_sem_num)

        debt_burden = df_unique.groupby(pay_col)[debt_cols].mean().reset_index()

        debt_burden_melted = debt_burden.melt(
            id_vars=pay_col,
            var_name='Колонка_Семестра',
            value_name='Среднее_число_долгов'
        )

        debt_burden_melted['Семестр'] = debt_burden_melted['Колонка_Семестра'].apply(
            lambda x: f"Сем {extract_sem_num(x)}")

        debt_burden_melted = debt_burden_melted.sort_values(by=['Семестр', pay_col])

        fig7 = px.bar(
            debt_burden_melted,
            x='Семестр',
            y='Среднее_число_долгов',
            color=pay_col,
            barmode='group',
            text_auto='.2f',
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={'Среднее_число_долгов': 'Среднее кол-во долгов на 1 чел.'}
        )

        fig7.update_layout(
            height=500,
            margin=dict(t=30, b=50, l=50, r=50),
            legend_title_text="Форма зачисления:",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig7.update_traces(
            textposition='outside',
            textfont=dict(size=11),
            cliponaxis=False
        )

        st.plotly_chart(fig7, use_container_width=True, theme="streamlit")

        try:
            total_debts_series = df_unique[debt_cols].sum(axis=1)

            overall_burden = total_debts_series.groupby(df_unique[pay_col]).mean()

            if not overall_burden.empty and overall_burden.max() >= 0:
                top_burden_cat = overall_burden.idxmax()
                top_burden_val = overall_burden.max()

                st.info(
                    f"**Аналитика:** В среднем за весь период обучения наибольшую нагрузку в виде пересдач генерирует категория **«{top_burden_cat}»** "
                    f"(около **{top_burden_val:.2f}** накопленных долгов на одного студента).")
            else:
                st.write("Недостаточно данных для формирования текстового вывода.")

        except Exception as e:
            st.warning(f"Не удалось рассчитать текстовую аналитику. Детали: {e}")

except Exception as e:
    st.error(f"Не удалось рассчитать индекс нагрузки: {e}")

st.divider()

import pandas as pd
import plotly.graph_objects as go
import re

# ЭФФЕКТ СНЕЖНОГО КОМА
st.subheader("Эффект «Снежного кома»")
try:
    df_unique = df.drop_duplicates(subset=['ИД']).copy()

    all_debt_cols = [c for c in df_unique.columns if
                     'Несдано_вовремя_сем_' in c or 'Несдано_бинарных_вовремя_сем_' in c]

    def get_num(s):
        return int(re.search(r'\d+', s).group())

    all_debt_cols = sorted(all_debt_cols, key=get_num)

    max_sem = len(all_debt_cols)

    def categorize_debt(x):
        if pd.isna(x): return 'Отчислен/Нет'
        x = float(x)
        if x == 0:
            return '0 долгов'
        elif x == 1:
            return '1 долг'
        else:
            return '2+ долга'


    for i, col_name in enumerate(all_debt_cols, start=1):
        df_unique[f'Долги_Сем{i}'] = df_unique[col_name].apply(categorize_debt)

    node_dict = {}
    node_counter = 0
    sources, targets, values, link_colors = [], [], [], []

    color_map = {
        '0 долгов': 'rgba(100, 149, 237, 0.4)',
        '1 долг': 'rgba(255, 165, 0, 0.4)',
        '2+ долга': 'rgba(255, 69, 0, 0.4)',
        'Отчислен/Нет': 'rgba(128, 128, 128, 0.2)'
    }

    for i in range(1, max_sem):
        col_from = f'Долги_Сем{i}'
        col_to = f'Долги_Сем{i + 1}'

        flow = df_unique.groupby([col_from, col_to]).size().reset_index(name='Количество')

        for _, row in flow.iterrows():
            val_from = row[col_from]
            val_to = row[col_to]
            count = row['Количество']

            if count == 0: continue

            node_from_name = f"{val_from} (Сем {i})"
            node_to_name = f"{val_to} (Сем {i + 1})"

            if node_from_name not in node_dict:
                node_dict[node_from_name] = node_counter
                node_counter += 1
            if node_to_name not in node_dict:
                node_dict[node_to_name] = node_counter
                node_counter += 1

            sources.append(node_dict[node_from_name])
            targets.append(node_dict[node_to_name])
            values.append(count)
            link_colors.append(color_map.get(val_from, 'rgba(200,200,200,0.2)'))

    labels = [None] * len(node_dict)
    for name, idx in node_dict.items():
        labels[idx] = name

    node_colors = []
    for label in labels:
        if '0 долгов' in label:
            node_colors.append('#6495ED')
        elif '1 долг' in label:
            node_colors.append('#FFA500')
        elif '2+ долга' in label:
            node_colors.append('#FF4500')
        else:
            node_colors.append('#808080')

    fig3 = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])

    fig3.update_layout(
        title_text="Визуализация динамики задолженностей студентов и их переноса в следующие семестры",
        font_size=11,
        height=800,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig3, use_container_width=True, theme="streamlit")

    st.info("**Аналитика:** Большое количество долгов c наибольшим риском влияет на отчисление при переходе с 1 на 2 семестр")

except Exception as e:
    st.error(f"Ошибка при расчете Sankey: {e}")