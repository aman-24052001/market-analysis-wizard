import os
import json
import logging
import re
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import io

from dotenv import load_dotenv
import google.generativeai as genai
from serpapi import GoogleSearch

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_research.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


@dataclass
class MarketResearchReport:
    """Data class to store market research results."""
    business_idea: str
    market_overview: str
    target_audience: Dict
    competitor_analysis: List[Dict]
    swot_analysis: Dict
    market_size: str
    market_trends: List[Dict]  # New field for market trends data
    recommendations: List[str]
    resource_requirements: str
    optimal_location: str
    references: List[str]
    timestamp: str = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)


class MarketResearchException(Exception):
    """Custom exception for market research errors."""
    pass


class MarketResearchTool:
    def __init__(self):
        """Initialize the Market Analysis Wizard and setup Gemini API."""
        self.setup_gemini_api()

    def setup_gemini_api(self) -> None:
        """Setup the Gemini API using the key from .env."""
        try:
            if not GEMINI_API_KEY:
                raise MarketResearchException("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Successfully initialized Gemini API")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise MarketResearchException(f"API initialization failed: {str(e)}")

    def fetch_competitor_news(self, query: str) -> str:
        """Use SerpAPI to search for competitor news and return a summary string with fallback."""
        try:
            params = {
                "engine": "google",
                "q": query,
                "hl": "en",
                "gl": "us",
                "google_domain": "google.com",
                "num": "10",
                "safe": "active",
                "api_key": SERPAPI_API_KEY
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            news_summaries = []
            for item in organic_results:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                source = item.get("displayed_link", "")
                news_summaries.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\nSource: {source}")
            summary_text = "\n\n".join(news_summaries)
            return summary_text if summary_text else "No competitor news available."
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return "No competitor news available."

    def _create_enhanced_prompt(self, business_idea: str, news_summary: str) -> str:
        """Create an enhanced prompt including competitor news analysis and additional sections."""
        prompt = f"""
You are an expert market research analyst. Analyze the following business idea and provide a detailed market research report in valid JSON format without any comments or extra text. Your response should be written in a detailed, professional, human tone—as if prepared by a seasoned analyst.

Business Idea: "{business_idea}"

Additionally, incorporate the following competitor news and market trend information (fetched using SerpAPI):
"{news_summary}"

Your analysis should be in human tone showing professional and indepth research & must include:
- A detailed market overview.
- A target audience breakdown including demographics and psychographics(talk in %).
- Competitor analysis with competitor names, market share (as a percentage), and recent activities.
- A SWOT analysis (strengths, weaknesses, opportunities, threats) in one single line for each.
- An assessment of market size and potential.
- Market trends data for the last 6 months showing month-wise market size/growth.
- Clear recommendations for the business.
- An estimation of the resource requirements to open the business(captital in rupees and land and human resource cost etc..).
- A recommendation on the optimal geographic location where the business is most likely to succeed.

The final JSON must strictly follow this structure (no comments allowed):
{{
    "market_overview": "...",
    "target_audience": {{
        "primary_demographic": "...",
        "psychographic_profile": "...",
        "pain_points": ["...", "..."],
        "buying_behavior": "..."
    }},
    "competitor_analysis": [
        {{
            "competitor_name": "...",
            "market_share": "...",
            "recent_activities": "..."
        }}
    ],
    "swot_analysis": {{
        "strengths": ["..."],
        "weaknesses": ["..."],
        "opportunities": ["..."],
        "threats": ["..."]
    }},
    "market_size": "...",
    "market_trends": [
        {{
            "month": "...",
            "market_size": "...",
            "growth_rate": "..."
        }}
    ],
    "recommendations": ["...", "..."],
    "resource_requirements": "...",
    "optimal_location": "...",
    "references": ["Source 1", "Source 2", "SerpAPI"]
}}
"""
        return prompt

    def _remove_comments(self, text: str) -> str:
        """Remove any lines starting with '//' or inline '//' comments."""
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith("//"):
                continue
            cleaned_line = re.sub(r'//.*', '', line)
            cleaned_lines.append(cleaned_line)
        return "\n".join(cleaned_lines)

    def _fix_invalid_parentheses(self, text: str) -> str:
        """Remove extraneous ' (SerpAPI)' from the response text."""
        fixed_text = re.sub(r'\s+\(SerpAPI\)', '', text)
        return fixed_text

    def _sanitize_response(self, text: str) -> str:
        """Remove control characters that are invalid in JSON."""
        sanitized = re.sub(r'[\x00-\x1F\x7F]', '', text)
        return sanitized

    def _parse_json_response(self, response: str) -> Dict:
        """Parse the JSON response after preprocessing."""
        try:
            fixed_response = self._fix_invalid_parentheses(response)
            sanitized_response = self._sanitize_response(fixed_response)
            return json.loads(sanitized_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}\nResponse: {response}")
            raise MarketResearchException(f"Failed to parse JSON: {str(e)}")

    def _validate_research_data(self, data: Dict) -> None:
        """Ensure all required fields are present in the response."""
        required_fields = [
            "market_overview", "target_audience", "competitor_analysis",
            "swot_analysis", "market_size", "market_trends", "recommendations",
            "resource_requirements", "optimal_location"
        ]
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise MarketResearchException(f"Missing fields in research data: {missing}")

    def analyze_business_idea(self, business_idea: str) -> MarketResearchReport:
        """Analyze a business idea and generate a market research report."""
        try:
            logger.info(f"Starting analysis for business idea: {business_idea}")

            news_query = f"competitor analysis {business_idea}"
            news_summary = self.fetch_competitor_news(news_query)
            logger.info("Fetched competitor news.")

            prompt = self._create_enhanced_prompt(business_idea, news_summary)
            logger.debug("Enhanced prompt created.")

            response = self.model.generate_content(prompt)
            if not response.text:
                raise MarketResearchException("Received empty response from Gemini API")
            logger.debug("Received response from Gemini API.")

            cleaned_response = self._remove_comments(response.text)
            cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
            logger.debug("Cleaned response from Gemini API:")
            logger.debug(cleaned_response)

            research_data = self._parse_json_response(cleaned_response)
            self._validate_research_data(research_data)

            report = MarketResearchReport(
                business_idea=business_idea,
                market_overview=research_data["market_overview"],
                target_audience=research_data["target_audience"],
                competitor_analysis=research_data["competitor_analysis"],
                swot_analysis=research_data["swot_analysis"],
                market_size=research_data["market_size"],
                market_trends=research_data["market_trends"],
                recommendations=research_data["recommendations"],
                resource_requirements=research_data["resource_requirements"],
                optimal_location=research_data["optimal_location"],
                references=research_data.get("references", ["SerpAPI"])
            )
            logger.info("Successfully generated market research report")
            return report
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}\n{traceback.format_exc()}")
            raise MarketResearchException(f"Analysis failed: {str(e)}")


def generate_pie_chart(competitor_analysis: List[Dict]) -> io.BytesIO:
    """Generate a pie chart showing competitor market share distribution."""
    try:
        try:
            plt.style.use('seaborn-darkgrid')
        except Exception:
            plt.style.use('ggplot')
    except Exception as e:
        logger.error(f"Error setting style: {str(e)}")

    labels = []
    sizes = []
    for comp in competitor_analysis:
        labels.append(comp.get("competitor_name", "Unknown"))
        try:
            percent_str = comp.get("market_share", "0%").strip('%')
            percent = float(percent_str) if percent_str.replace('.', '', 1).isdigit() else 0.0
        except:
            percent = 0.0
        sizes.append(percent)

    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Competitor Market Share Distribution", fontsize=12)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def generate_line_chart(market_trends: List[Dict]) -> io.BytesIO:
    """Generate a line chart for market trends using actual data."""
    try:
        try:
            plt.style.use('seaborn-darkgrid')
        except Exception:
            plt.style.use('ggplot')
    except Exception as e:
        logger.error(f"Error setting style: {str(e)}")

    # Extract data from market trends
    months = []
    growth_rates = []
    market_sizes = []

    for trend in market_trends:
        months.append(trend.get("month", ""))
        # Convert growth rate string to float
        growth_str = trend.get("growth_rate", "0%").strip('%')
        growth_rates.append(float(growth_str) if growth_str.replace('.', '', 1).isdigit() else 0.0)
        # Extract market size (assuming it's in numerical format)
        size_str = trend.get("market_size", "0")
        # Remove any non-numeric characters except decimal points
        size_str = re.sub(r'[^0-9.]', '', size_str)
        market_sizes.append(float(size_str) if size_str else 0.0)

    plt.figure(figsize=(8, 4))

    # Create subplot for dual axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot market size on primary axis
    line1 = ax1.plot(months, market_sizes, 'b-', marker='o', label='Market Size')
    ax1.set_xlabel("Month", fontsize=10)
    ax1.set_ylabel("Market Size", color='b', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot growth rate on secondary axis
    line2 = ax2.plot(months, growth_rates, 'r--', marker='s', label='Growth Rate (%)')
    ax2.set_ylabel("Growth Rate (%)", color='r', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper left')

    plt.title("Market Size and Growth Trends", fontsize=12)
    plt.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def create_pdf_report(report: MarketResearchReport) -> str:
    try:
        safe_idea = re.sub(r'[<>:"/\\|?*]', '', report.business_idea).replace(" ", "_").lower()[:50]
        filename = f"market_analysis_wizard_{safe_idea}_{datetime.now().strftime('%Y%m%d')}.pdf"
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            textColor=colors.darkblue,
            fontSize=26,
            spaceAfter=20
        )
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            alignment=TA_LEFT,
            textColor=colors.darkred,
            fontSize=16,
            spaceBefore=15,
            spaceAfter=10
        )
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=9,
            leading=12
        )
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )

        story = []
        # Title and Business Idea
        story.append(Paragraph("Market Analysis Wizard", title_style))
        story.append(Paragraph(f"Business Idea: {report.business_idea}", heading_style))
        story.append(Spacer(1, 12))

        # Market Overview
        story.append(Paragraph("1. Market Overview", heading_style))
        story.append(Paragraph(report.market_overview, normal_style))
        story.append(Spacer(1, 12))

        # Target Audience
        story.append(Paragraph("2. Target Audience", heading_style))
        ta = report.target_audience
        ta_text = f"""
        <b>Primary Demographic:</b> {ta.get('primary_demographic', '')}<br/>
        <b>Psychographic Profile:</b> {ta.get('psychographic_profile', '')}<br/>
        <b>Pain Points:</b> {', '.join(ta.get('pain_points', []))}<br/>
        <b>Buying Behavior:</b> {ta.get('buying_behavior', '')}
        """
        story.append(Paragraph(ta_text, normal_style))
        story.append(Spacer(1, 12))

        # Competitor Analysis with Pie Chart
        story.append(Paragraph("3. Competitor Analysis", heading_style))
        comp_text = ""
        for comp in report.competitor_analysis:
            comp_text += f"<b>{comp.get('competitor_name', 'Unknown')}</b> (Market Share: {comp.get('market_share', 'N/A')})<br/>"
            comp_text += f"Recent Activities: {comp.get('recent_activities', '')}<br/><br/>"
        story.append(Paragraph(comp_text, normal_style))

        # Add pie chart
        pie_chart_buf = generate_pie_chart(report.competitor_analysis)
        pie_img = RLImage(pie_chart_buf, width=4 * inch, height=4 * inch)
        story.append(pie_img)
        story.append(Spacer(1, 12))

        # SWOT Analysis
        story.append(Paragraph("4. SWOT Analysis", heading_style))
        swot = report.swot_analysis

        # Function to wrap text
        def wrap_text(text_list, max_chars=40):
            wrapped_text = []
            for item in text_list:
                words = item.split()
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 <= max_chars:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        wrapped_text.append(" ".join(current_line))
                        current_line = [word]
                        current_length = len(word)

                if current_line:
                    wrapped_text.append(" ".join(current_line))

            return "\n".join(wrapped_text)

        # SWOT table data
        data = [
            ['Strengths', 'Weaknesses'],
            [wrap_text(swot.get('strengths', [])), wrap_text(swot.get('weaknesses', []))],
            ['Opportunities', 'Threats'],
            [wrap_text(swot.get('opportunities', [])), wrap_text(swot.get('threats', []))]
        ]

        # SWOT table style
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('BACKGROUND', (0, 2), (-1, 2), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 2), (-1, 2), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, 1), colors.lavender),
            ('BACKGROUND', (0, 3), (-1, 3), colors.lavender),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('WORDWRAP', (0, 0), (-1, -1)),
        ])

        swot_table = Table(data, colWidths=[3 * inch, 3 * inch], rowHeights=[20, None, 20, None])
        swot_table.setStyle(table_style)
        story.append(KeepTogether(swot_table))
        story.append(Spacer(1, 12))

        # Market Size and Trends
        story.append(Paragraph("5. Market Size and Trends", heading_style))
        story.append(Paragraph(report.market_size, normal_style))
        story.append(Spacer(1, 12))

        # Add line chart for market trends
        line_chart_buf = generate_line_chart(report.market_trends)
        line_img = RLImage(line_chart_buf, width=6 * inch, height=4 * inch)
        story.append(line_img)
        story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph("6. Recommendations", heading_style))
        rec_text = "<br/>".join([f"• {rec}" for rec in report.recommendations])
        story.append(Paragraph(rec_text, normal_style))
        story.append(Spacer(1, 12))

        # Resource Requirements
        story.append(Paragraph("7. Resource Requirements", heading_style))
        story.append(Paragraph(report.resource_requirements, normal_style))
        story.append(Spacer(1, 12))

        # Optimal Location
        story.append(Paragraph("8. Optimal Location", heading_style))
        story.append(Paragraph(report.optimal_location, normal_style))
        story.append(Spacer(1, 12))

        # References
        story.append(Paragraph("9. References", heading_style))
        ref_text = "<br/>".join([f"• {ref}" for ref in report.references])
        story.append(Paragraph(ref_text, normal_style))

        # Footer with timestamp
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated on: {report.timestamp}", footer_style))

        doc.build(story)
        return filename
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}\n{traceback.format_exc()}")
        raise MarketResearchException(f"PDF generation failed: {str(e)}")


def main():
    try:
        tool = MarketResearchTool()
        business_idea = input("Enter your business idea: ")
        report = tool.analyze_business_idea(business_idea)

        # Print the JSON output for verification
        print("\nMarket Research Report (JSON):")
        print(json.dumps(report.to_dict(), indent=4, ensure_ascii=False))

        pdf_filename = create_pdf_report(report)
        logger.info("Market research analysis completed successfully.")
        logger.info(f"PDF report saved as: {pdf_filename}")
        print("\nPDF report generated successfully!")
        print(f"PDF report: {pdf_filename}")
    except MarketResearchException as e:
        logger.error(f"Market research failed: {str(e)}")
        print(f"\nError: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        print("\nUnexpected error occurred. Check market_research.log for details.")


if __name__ == "__main__":
    main()
