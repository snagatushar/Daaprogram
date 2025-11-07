// src/App.js
import React, { useState } from "react";

/*
  Single-file React app (no external CSS).
  - Paste this into src/App.js (replace existing content)
  - Run with: npm start
*/

const programs = [
  {
    id: "a",
    title: "Independent t-test (Two-sample)",
    filename: "independent_ttest.py",
    code: `import numpy as np
from scipy import stats
# Sample data for two independent groups
group_a = np.array([25, 28, 30, 26, 29, 27, 31, 24, 28, 26])
group_b = np.array([22, 24, 23, 20, 25, 21, 22, 23, 20, 24])
# Perform independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
alpha = 0.05
if p_value <= alpha:
    print("Conclusion: Reject the null hypothesis. There is a significant difference between the group means.")
else:
    print("Conclusion: Fail to reject the null hypothesis. There is no significant difference between the group means.")`,
  },
  {
    id: "b",
    title: "One-sample t-test",
    filename: "one_sample_ttest.py",
    code: `import numpy as np
from scipy import stats
sample_data = np.array([650, 470, 520, 480, 530, 450, 560, 480, 510])
hypothesized_mean = 500
t_statistic, p_value = stats.ttest_1samp(sample_data, popmean=hypothesized_mean)
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
alpha = 0.05
if p_value <= alpha:
    print("Conclusion: Reject the null hypothesis. There is a significant difference between the group means.")
else:
    print("Conclusion: Fail to reject the null hypothesis. There is no significant difference between the group means.")`,
  },
  {
    id: "c",
    title: "One-way ANOVA",
    filename: "one_way_anova.py",
    code: `from scipy.stats import f_oneway
import numpy as np
# Example data for three groups
group1 = np.array([10, 12, 11, 13, 10])
group2 = np.array([15, 14, 16, 17, 15])
group3 = np.array([18, 19, 17, 20, 18])
# Perform one-way ANOVA
f_statistic, p_value = f_oneway(group1, group2, group3)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference between group means.")
else:
    print("Fail to reject the null hypothesis: No significant difference between group means.")`,
  },
  {
    id: "d",
    title: "Two-way ANOVA",
    filename: "two_way_anova.py",
    code: `import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
# Example dataset
data = {
    'score': [23, 25, 27, 22, 30, 34, 31, 40, 42, 39, 41],
    'treatment': ['A','A','A','B','B','B','B','C','C','C','C'],
    'gender': ['M','F','M','F','M','F','M','F','M','F','F']
}
df = pd.DataFrame(data)
# Two-way ANOVA
model = ols('score ~ C(treatment) + C(gender) + C(treatment):C(gender)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)`,
  },
  {
    id: "e",
    title: "Chi-square Test (Heatmap)",
    filename: "chi_square_heatmap.py",
    code: `import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
# Load dataset
df = pd.read_csv("fraud_detection_hypothesis_testing.csv")
# Suppose we have categorical columns: region and is_fraud
if 'region' not in df.columns:
    import numpy as np
    np.random.seed(0)
    df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(df))
# Create contingency table
contingency = pd.crosstab(df['region'], df['is_fraud'])
print("\\nContingency Table:\\n", contingency)
# Perform Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency)
print("\\n=== Chi-square Test ===")
print(f"Chi-square statistic = {chi2:.4f}")
print(f"Degrees of freedom = {dof}")
print(f"p-value = {p:.6f}")
# Visualization
plt.figure(figsize=(7,5))
sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Chi-square Test Visualization: Region vs Fraud")
plt.xlabel("Is Fraud (0 = No, 1 = Yes)")
plt.ylabel("Region")
plt.show()` ,
  },
  {
    id: "f",
    title: "Simple Linear Regression",
    filename: "simple_linear_regression.py",
    code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# Load dataset
df = pd.read_csv("fraud_detection_hypothesis_testing.csv")
if 'hour' not in df.columns:
    np.random.seed(0)
    df['hour'] = np.random.randint(0, 24, size=len(df))
X = df[['hour']]
Y = df['amount']
model = LinearRegression()
model.fit(X, Y)
intercept = model.intercept_
slope = model.coef_[0]
print(f"Fitted Model: Y = {intercept:.2f} + {slope:.2f}X")
Y_pred = model.predict(X)
r2 = r2_score(Y, Y_pred)
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
plt.figure(figsize=(7,5))
sns.scatterplot(x='hour', y='amount', data=df, alpha=0.6)
plt.plot(df['hour'], Y_pred, color='red')
plt.xlabel('Hour')
plt.ylabel('Transaction Amount')
plt.title('Simple Linear Regression: Amount vs Hour')
plt.show()` ,
  },
  {
    id: "g",
    title: "Multiple Linear Regression",
    filename: "multiple_linear_regression.py",
    code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# Load dataset
df = pd.read_csv("fraud_detection_hypothesis_testing.csv")
if 'hour' not in df.columns:
    np.random.seed(0)
    df['hour'] = np.random.randint(0, 24, size=len(df))
X = df[['hour', 'is_fraud']]
y = df['amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
new_data = pd.DataFrame({'hour': [10, 15], 'is_fraud': [0, 1]})
predicted_amount = model.predict(new_data)
print(pd.concat([new_data, pd.Series(predicted_amount, name='Predicted_Amount')], axis=1))` ,
  },
  {
    id: "h",
    title: "Logistic Regression",
    filename: "logistic_regression.py",
    code: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load dataset
df = pd.read_csv("fraud_detection_hypothesis_testing.csv")
if 'hour' not in df.columns:
    import numpy as np
    np.random.seed(0)
    df['hour'] = np.random.randint(0, 24, size=len(df))
X = df[['hour', 'amount']]
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")
print(cm)
print(classification_report(y_test, y_pred))
new_data = pd.DataFrame({'hour': [10, 15], 'amount': [5000, 20000]})
predictions = model.predict(new_data)
print(pd.concat([new_data, pd.Series(predictions, name='Predicted_is_fraud')], axis=1))` ,
  },
  {
    id: "i",
    title: "Fraud Detection - Hypothesis Testing & CI",
    filename: "fraud_hypothesis_testing.py",
    code: `import pandas as pd
import numpy as np
from scipy import stats
# Load the dataset
df = pd.read_csv("fraud_detection_hypothesis_testing.csv")
fraud = df[df['is_fraud'] == 1]['amount']
nonfraud = df[df['is_fraud'] == 0]['amount']
# Welch's t-test
t_stat, p_two_sided = stats.ttest_ind(fraud, nonfraud, equal_var=False)
# One-sided p-value
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
print("Mean (Fraud):", np.mean(fraud))
print("Mean (Non-Fraud):", np.mean(nonfraud))
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value (one-sided): {p_one_sided:.6f}")
alpha = 0.05
if p_one_sided < alpha:
    print("Fraud transactions have significantly higher mean amounts.")
else:
    print("No significant difference between fraud and non-fraud transaction means.")` ,
  },

  // New sections requested by you:
  {
    id: "powerbi",
    title: "Power BI – Steps",
    filename: "PowerBI_Steps.txt",
    code: `1️⃣ Load Dataset in Power BI

Open Power BI Desktop
Click Get Data → Text/CSV
Select file → Load

2️⃣ Create Slicer (Salesperson)

Go to Visualizations → select Slicer
Drag SalesPerson → Field
Place on left side of report

3️⃣ Create Bar Chart (Sum of SalesAmount by Product)

Select Clustered Column Chart
Drag Product → Axis
Drag SalesAmount → Values
Place on right side of report

4️⃣ Check Slicer Filter

Click any salesperson in slicer
Verify chart updates dynamically`,
  },
  {
    id: "excel",
    title: "Excel – Formulas & Features",
    filename: "Excel_Steps.txt",
    code: `5️⃣ Excel – Calculate Total Sales

In F2, type =D2*E2
Drag formula down

6️⃣ Excel – Conditional Formatting

Select F2:F7
Home → Conditional Formatting → Highlight Cell Rules → Greater Than
Enter 400000 → Choose Green Fill

7️⃣ Excel – Create Performance Column

In G1, type Performance
In G2, type =IF(F2>=400000,"High","Low")
Drag down

8️⃣ Excel – Count High Performers

In any empty cell, type =COUNTIF(G2:G7,"High")

9️⃣ Excel – Total Sales by Salesperson

In any empty cell, type =SUMIF(A2:A7,"Asha",F2:F7)`,
  },
  {
    id: "pivot",
    title: "Pivot Table & What-If",
    filename: "PivotTable_Steps.txt",
    code: `🔟 Excel – Create Pivot Table

Select A1:F7
Insert → PivotTable → New Worksheet → OK
Drag Salesperson → Rows
Drag Total Sales → Values

1️⃣1️⃣ Excel – What-If Analysis

Go to Data → What-If Analysis → Scenario Manager
Click Add
Select cells to change (e.g., Unit Price for Tablet)
Enter new values → OK → Show

1️⃣2️⃣ Excel – Data Validation (Dropdown List)

Select cell (e.g., H1)
Data → Data Validation
Allow: → select List
Source: → =A2:A7
Click OK

1️⃣3️⃣ Excel – Subtotals

Sort by Region
Data → Subtotal
At each change in: Region
Use function: Sum
Add subtotal to: Total Sales → OK`,
  },
];

function App() {
  const [selected, setSelected] = useState(programs[0].id);
  const [copiedId, setCopiedId] = useState(null);

  const handleCopy = async (code, id) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error("Copy failed", err);
      alert("Copy failed — check browser permissions.");
    }
  };

  const handleDownload = (code, filename) => {
    const blob = new Blob([code], { type: "text/plain" });
    const href = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = href;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(href);
  };

  // CSS injected here so no separate file is needed
  const css = `
    :root{
      --bg: #0f0f10;
      --panel: #1b1b1d;
      --muted: #9aa0a6;
      --accent: #6ee7b7;
    }
    *{box-sizing:border-box}
    body,html,#root{height:100%;margin:0}
    .app-root{
      min-height:100vh;
      background:var(--bg);
      color:#eaeaea;
      font-family: Inter, Roboto, Arial, sans-serif;
      padding:20px;
    }
    .header h1{margin:0;font-size:24px}
    .tagline{margin:4px 0 18px;color:var(--muted);font-size:13px}
    .layout{display:flex;gap:18px}
    .left{width:260px}
    .label{display:block;margin-bottom:6px;color:var(--muted);font-size:13px}
    .select{width:100%;padding:8px;background:#0b0b0b;border:1px solid #262626;color:#fff;border-radius:6px}
    .list{margin-top:12px;display:flex;flex-direction:column;gap:8px}
    .list-item{background:var(--panel);color:#eaeaea;border:1px solid #2a2a2c;padding:10px;text-align:left;border-radius:6px;cursor:pointer}
    .list-item.active{outline:2px solid rgba(110,231,183,0.12)}
    .main{flex:1}
    .card{background:var(--panel);border:1px solid #2a2a2c;border-radius:8px;margin-bottom:16px;overflow:hidden}
    .card.hidden{display:none}
    .card-header{display:flex;justify-content:space-between;align-items:center;padding:12px;border-bottom:1px solid #151515}
    .card-title{margin:0;font-size:16px}
    .filename{color:var(--muted);font-size:12px}
    .controls{display:flex;gap:8px;align-items:center}
    .mini-select{background:#0b0b0b;color:#fff;border:1px solid #262626;padding:6px;border-radius:6px}
    .copy{background:#0b0b0b;border:1px solid #333;padding:8px 10px;border-radius:6px;color:#eaeaea;cursor:pointer}
    .copy:hover{background:#161616}
    .card-body{padding:12px}
    .code{background:#060606;border-radius:6px;padding:12px;color:var(--accent);font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,monospace;font-size:13px;white-space:pre-wrap;overflow:auto;max-height:520px}
    .footer{margin-top:10px;color:var(--muted);font-size:12px}
    @media(max-width:860px){
      .layout{flex-direction:column}
      .left{width:100%}
    }
  `;

  return (
    <div className="app-root">
      <style>{css}</style>

      <header className="header">
        <h1>Stats & ML Playground</h1>
        <p className="tagline">Dark themed examples — copy or download any snippet.</p>
      </header>

      <div className="layout">
        <aside className="left">
          <label className="label">Choose program</label>
          <select
            className="select"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            {programs.map((p) => (
              <option key={p.id} value={p.id}>
                {p.title}
              </option>
            ))}
          </select>

          <div className="list">
            {programs.map((p) => (
              <button
                key={p.id}
                className={`list-item ${selected === p.id ? "active" : ""}`}
                onClick={() => setSelected(p.id)}
              >
                {p.title}
              </button>
            ))}
          </div>
        </aside>

        <main className="main">
          {programs.map((p) => (
            <section
              key={p.id}
              className={`card ${selected === p.id ? "visible" : "hidden"}`}
            >
              <div className="card-header">
                <div>
                  <h2 className="card-title">{p.title}</h2>
                  <div className="filename">{p.filename}</div>
                </div>

                <div className="controls">
                  <select
                    className="mini-select"
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === "show") setSelected(p.id);
                      if (v === "hide") setSelected("");
                      if (v === "download") handleDownload(p.code, p.filename);
                      e.target.value = "show";
                    }}
                    defaultValue="show"
                  >
                    <option value="show">Show</option>
                    <option value="hide">Hide</option>
                    <option value="download">Download .txt</option>
                  </select>

                  <button className="copy" onClick={() => handleCopy(p.code, p.id)}>
                    {copiedId === p.id ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>

              <div className="card-body">
                <pre className="code">
                  <code>{p.code}</code>
                </pre>
              </div>
            </section>
          ))}
        </main>
      </div>

      <footer className="footer">Made with ❤️ — paste the snippets into your editor or Excel/Power BI as needed.</footer>
    </div>
  );
}

export default App;
