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
    st.subheader("מדדי טיב המודל")
    st.write("המדדים העיקריים מחושבים על **נתוני הבדיקה (20%)** — דירות שהמודל לא ראה באימון.")
    m = pd.DataFrame({"אימון (80%)": res["train_metrics"], "בדיקה (20%)": res["test_metrics"]}).T
    t = res["test_metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² (בדיקה)", f"{t['R2']:.3f}")
    k2.metric("MAE (בדיקה)", f"${t['MAE']:,.0f}")
    k3.metric("RMSE (בדיקה)", f"${t['RMSE']:,.0f}")
    k4.metric("MAPE (בדיקה)", f"{t['MAPE']:.1f}%")
    st.dataframe(m.style.format({"R2": "{:.3f}", "MAE": "${:,.0f}",
                                 "RMSE": "${:,.0f}", "MAPE": "{:.1f}%"}))
    st.markdown(
        "- **R²** — חלק השונות במחיר שהמודל מסביר.\n"
        "- **MAE** — השגיאה הממוצעת בדולרים.\n"
        "- **RMSE** — שגיאה שמענישה יותר טעויות גדולות.\n"
        "- **MAPE** — השגיאה הממוצעת באחוזים ממחיר הדירה.\n\n"
        "הפער הקטן בין אימון לבדיקה מעיד שאין התאמת-יתר (overfitting) משמעותית.")

    tr = res["test_results"]
    fig = px.scatter(tr, x="actual", y="predicted", opacity=0.6,
                     labels={"actual": "מחיר בפועל", "predicted": "מחיר חזוי"},
                     title="מחיר חזוי מול מחיר בפועל — נתוני בדיקה")
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
