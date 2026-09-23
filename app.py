"""Streamlit app: apartment pricing with linear regression."""
import pandas as pd
import plotly.express as px
import streamlit as st

from model import CATEGORICAL, NUMERIC, encode, load_data, predict, train

st.set_page_config(page_title="תמחור דירות", page_icon="🏠", layout="wide")
st.markdown("<style>body, .stMarkdown, .stTabs {direction: rtl; text-align: right;}</style>",
            unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return train(get_data())


df = get_data()
res = get_model()

st.title("🏠 מערכת תמחור דירות — רגרסיה לינארית")
st.caption(f"{len(df):,} דירות · אימון על 80% ({res['n_train']:,}) · בדיקה על 20% ({res['n_test']:,})")

tab_pred, tab_eval, tab_model, tab_data = st.tabs(
    ["💰 תמחור דירה", "📊 הערכת ביצועים", "🧮 בניית המודל וקידוד", "📁 הנתונים"])

# ---------------- Prediction ----------------
with tab_pred:
    st.subheader("הזינו מאפייני דירה לקבלת מחיר חזוי")
    c1, c2, c3 = st.columns(3)
    with c1:
        bedrooms = st.number_input("חדרי שינה (bedrooms)", 0, 10, 3)
        bathrooms = st.number_input("חדרי רחצה (bathrooms)", 0.0, 8.0, 2.0, 0.25)
        floors = st.selectbox("מספר קומות (floors)", [1, 1.5, 2, 2.5, 3], index=0)
        yr_built = st.number_input("שנת בנייה (yr_built)", 1900, 2026, 1985)
    with c2:
        sqft_above = st.number_input("שטח מעל הקרקע (sqft_above)", 300, 10000, 1400, 50)
        sqft_basement = st.number_input("שטח מרתף (sqft_basement)", 0, 5000, 400, 50)
        sqft_lot = st.number_input("שטח המגרש (sqft_lot)", 500, 2_000_000, 6000, 100)
        sqft_living = sqft_above + sqft_basement
        st.info(f"שטח מגורים כולל (sqft_living) = {sqft_living:,}")
    with c3:
        waterfront = st.radio("נוף לחוף מים (waterfront)", [0, 1],
                              format_func=lambda v: "כן" if v else "לא", horizontal=True)
        view = st.slider("איכות הנוף (view) 0–4", 0, 4, 0)
        condition = st.slider("מצב הדירה (condition) 1–5", 1, 5, 3)

    features = dict(bedrooms=bedrooms, bathrooms=bathrooms, sqft_living=sqft_living,
                    sqft_lot=sqft_lot, floors=floors, waterfront=waterfront, view=view,
                    condition=condition, sqft_above=sqft_above,
                    sqft_basement=sqft_basement, yr_built=yr_built)
    price = predict(res, features)
    mae = res["test_metrics"]["MAE"]

    st.metric("מחיר חזוי", f"${price:,.0f}")
    st.caption(f"טווח שגיאה טיפוסי (±MAE על נתוני הבדיקה): "
               f"${max(price - mae, 0):,.0f} – ${price + mae:,.0f}")

    similar = df[(df.sqft_living.between(sqft_living * 0.85, sqft_living * 1.15))
                 & (df.bedrooms == bedrooms)]
    if len(similar):
        st.write(f"**בדיקת סבירות:** נמצאו {len(similar)} דירות דומות בנתונים "
                 f"(שטח ±15%, אותו מספר חדרי שינה). מחיר חציוני: "
                 f"**${similar.price.median():,.0f}** "
                 f"(טווח ${similar.price.quantile(.25):,.0f} – ${similar.price.quantile(.75):,.0f}).")
    if price <= 0:
        st.warning("המודל החזיר מחיר לא הגיוני — המאפיינים רחוקים מטווח הנתונים שעליהם אומן המודל.")

# ---------------- Evaluation ----------------
with tab_eval:
    st.subheader("הערכת טיב המודל — R-Squared (R²)")
    st.write("מדד ההערכה של המודל הוא **R²**, והוא מחושב על **נתוני הבדיקה (20%)** — "
             "דירות שהמודל לא ראה באימון.")
    t, tr_m = res["test_metrics"], res["train_metrics"]
    k1, k2, k3 = st.columns(3)
    k1.metric("R² — נתוני בדיקה (20%)", f"{t['R2']:.3f}",
              delta=f"{t['R2'] - tr_m['R2']:+.3f} לעומת אימון", delta_color="off")
    k2.metric("R² — נתוני אימון (80%)", f"{tr_m['R2']:.3f}")
    k3.metric("R² מתוקנן (Adjusted) — בדיקה", f"{t['Adj_R2']:.3f}")
    st.progress(max(min(t["R2"], 1.0), 0.0),
                text=f"המודל מסביר {t['R2']:.1%} מהשונות במחירי הדירות בנתוני הבדיקה")

    st.latex(r"R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}")
    st.markdown(
        f"- **מה המדד אומר:** R² מודד איזה חלק מהשונות במחיר הדירות מוסבר על ידי המודל. "
        f"1 = חיזוי מושלם, 0 = המודל לא טוב יותר מניחוש המחיר הממוצע.\n"
        f"- **התוצאה:** R² = **{t['R2']:.3f}** על נתוני הבדיקה — המודל מסביר כ-"
        f"{t['R2']:.0%} מהשונות במחיר. זהו כוח הסבר בינוני-טוב עבור מודל לינארי פשוט "
        f"ללא משתני מיקום.\n"
        f"- **אימון מול בדיקה:** {tr_m['R2']:.3f} באימון לעומת {t['R2']:.3f} בבדיקה — "
        f"ירידה קטנה, כלומר אין התאמת-יתר (overfitting) משמעותית והמודל מכליל לדירות חדשות.\n"
        f"- **R² מתוקנן ({t['Adj_R2']:.3f}):** מתקן את R² לפי מספר המשתנים במודל "
        f"({len(res['columns'])}), כך שהוספת משתנים לא-רלוונטיים לא תנפח את המדד.")

    with st.expander("מדדים משלימים (MAE, RMSE, MAPE)"):
        m = pd.DataFrame({"אימון (80%)": tr_m, "בדיקה (20%)": t}).T
        st.dataframe(m.style.format({"R2": "{:.3f}", "Adj_R2": "{:.3f}", "MAE": "${:,.0f}",
                                     "RMSE": "${:,.0f}", "MAPE": "{:.1f}%"}))

    tr = res["test_results"]
    fig = px.scatter(tr, x="actual", y="predicted", opacity=0.6,
                     labels={"actual": "מחיר בפועל", "predicted": "מחיר חזוי"},
                     title=f"מחיר חזוי מול מחיר בפועל — נתוני בדיקה (R² = {t['R2']:.3f})")
    lim = [0, max(tr.actual.max(), tr.predicted.max())]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="חיזוי מושלם",
                    line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig, width="stretch")

    tr = tr.assign(residual=tr.actual - tr.predicted)
    st.plotly_chart(px.histogram(tr, x="residual", nbins=60,
                                 title="התפלגות השאריות (בפועל − חזוי)",
                                 labels={"residual": "שארית ($)"}),
                    width="stretch")

# ---------------- Model & encoding ----------------
with tab_model:
    st.subheader("קידוד משתנים")
    st.markdown(
        f"- **waterfront** — משתנה דמי (0/1), נכנס כמו שהוא.\n"
        f"- **view (0–4)** ו-**condition (1–5)** — משתנים קטגוריאליים (אינדקסים). "
        f"קודדו ב-**One-Hot Encoding** עם `drop_first` כדי להימנע ממלכודת הדמי: "
        f"קטגוריית הבסיס היא view=0 ו-condition=1.\n"
        f"- **sqft_above** — הוסר, כי `sqft_living = sqft_above + sqft_basement` בדיוק "
        f"(מולטיקוליניאריות מושלמת).\n"
        f"- משתנים מספריים: {', '.join(NUMERIC)}.")
    st.write("דוגמה — 5 השורות הראשונות אחרי קידוד:")
    st.dataframe(encode(df.head()))

    st.subheader("מקדמי הרגרסיה")
    st.write(f"חותך (intercept): **{res['intercept']:,.0f}**")
    coef = res["coefficients"].rename("מקדם").to_frame()
    st.dataframe(coef.style.format("{:,.1f}"))
    st.plotly_chart(px.bar(coef.reset_index(), x="מקדם", y="index", orientation="h",
                           labels={"index": "משתנה"}, title="השפעת כל משתנה על המחיר ($)"),
                    width="stretch")
    st.caption("מקדם של משתנה דמי (למשל view_4) = תוספת המחיר לעומת קטגוריית הבסיס, "
               "כשכל שאר המשתנים קבועים.")

# ---------------- Data ----------------
with tab_data:
    st.subheader("נתוני הדירות")
    st.dataframe(df, width="stretch")
    st.dataframe(df.describe().T)
    st.plotly_chart(px.scatter(df, x="sqft_living", y="price", color="view", opacity=0.6,
                               title="מחיר מול שטח מגורים"), width="stretch")
