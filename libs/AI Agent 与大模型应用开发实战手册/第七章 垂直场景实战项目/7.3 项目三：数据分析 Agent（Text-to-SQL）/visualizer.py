import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from enum import Enum
from pydantic import BaseModel, Field
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    TABLE = "table"


class ChartDecision(BaseModel):
    chart_type: ChartType
    x_column: str = Field(description="X 轴或标签列的列名")
    y_column: str = Field(description="Y 轴或数值列的列名")
    color_column: str | None = Field(None, description="分组/颜色列")
    reasoning: str = Field(description="选择此图表类型的理由")
    title: str = Field(description="图表标题")


def _infer_chart_type(question: str, df: pd.DataFrame) -> ChartDecision:
    col_info = {
        col: {
            "dtype": str(df[col].dtype),
            "sample": df[col].head(3).tolist(),
            "unique_count": int(df[col].nunique()),
        }
        for col in df.columns
    }

    prompt = f"""你是数据可视化专家，分析以下数据并选择最合适的图表类型。

**用户原始问题：** {question}

**数据结构（共 {len(df)} 行，{len(df.columns)} 列）：**
{json.dumps(col_info, ensure_ascii=False, indent=2)}

**图表选择规则：**
- 如果有时间/日期列（dtype 为 object 且值格式如 2024-01、周一） → line
- 如果是分类列（unique_count <= 20）对应数值列的对比 → bar
- 如果有"占比"、"比例"、"份额"等语义，且分类 <= 10 → pie
- 如果两个数值列之间有相关性分析意图 → scatter
- 其他情况 → table

**重要：x_column 和 y_column 必须是数据中实际存在的列名，区分大小写。**
"""

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=ChartDecision,
        temperature=0,
    )
    return response.choices[0].message.parsed


def _generate_summary(question: str, sql: str, df: pd.DataFrame, chart_type: ChartType) -> str:
    summary_stats = df.describe(include="all").to_string() if len(df) <= 20 else (
        f"共 {len(df)} 行数据。"
        f"主要数值列统计：{df.select_dtypes(include='number').describe().to_string()}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"查询 SQL：{sql}\n\n"
                f"数据统计：\n{summary_stats}\n\n"
                "请用 2-3 句话提炼核心业务洞察，语言面向业务人员，"
                "直接说明最重要的发现，不要重复问题，不要提 SQL。"
            )
        }],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


class DataVisualizer:
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def visualize(self, question: str, sql: str, df: pd.DataFrame, save_html: bool = True) -> dict:
        decision = _infer_chart_type(question, df)
        print(f"📊 图表决策：{decision.chart_type.value} — {decision.reasoning}")

        fig = self._render_chart(df, decision)

        html_path = None
        if save_html and fig is not None:
            safe_title = "".join(
                c for c in decision.title if c.isalnum() or c in "_- "
            )[:50]
            html_path = str(self.output_dir / f"{safe_title}.html")
            fig.write_html(html_path)
            print(f"💾 图表已保存：{html_path}")

        summary = _generate_summary(question, sql, df, decision.chart_type)
        print(f"\n📝 数据摘要：\n{summary}")

        return {
            "chart_type": decision.chart_type.value,
            "title": decision.title,
            "html_path": html_path,
            "figure": fig,
            "summary": summary,
            "decision": decision,
        }

    def _render_chart(self, df: pd.DataFrame, decision: ChartDecision) -> go.Figure | None:
        required_cols = [decision.x_column, decision.y_column]
        for col in required_cols:
            if col not in df.columns:
                matched = [c for c in df.columns if c.lower() == col.lower()]
                if matched:
                    if col == decision.x_column:
                        decision.x_column = matched[0]
                    else:
                        decision.y_column = matched[0]
                else:
                    print(f"⚠️  列名 '{col}' 不存在")
                    decision.chart_type = ChartType.TABLE

        common_kwargs = dict(
            title=decision.title,
            template="plotly_white",
        )

        if decision.chart_type == ChartType.BAR:
            fig = px.bar(
                df,
                x=decision.x_column,
                y=decision.y_column,
                color=decision.color_column,
                text_auto=".2s",
                **common_kwargs,
            )
            fig.update_layout(xaxis_tickangle=-30)

        elif decision.chart_type == ChartType.LINE:
            fig = px.line(
                df,
                x=decision.x_column,
                y=decision.y_column,
                color=decision.color_column,
                markers=True,
                **common_kwargs,
            )

        elif decision.chart_type == ChartType.PIE:
            fig = px.pie(
                df,
                names=decision.x_column,
                values=decision.y_column,
                hole=0.35,
                **common_kwargs,
            )

        elif decision.chart_type == ChartType.SCATTER:
            fig = px.scatter(
                df,
                x=decision.x_column,
                y=decision.y_column,
                color=decision.color_column,
                trendline="ols",
                **common_kwargs,
            )

        elif decision.chart_type == ChartType.TABLE:
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns), fill_color='paleturquoise'),
                cells=dict(values=[df[col] for col in df.columns], fill_color='lavender')
            )])
            fig.update_layout(title=decision.title)

        else:
            fig = None

        return fig