# Market Analysis Wizard 🚀

A powerful Python-based tool that generates comprehensive market research reports using AI and real-time data analysis. The tool leverages Google's Gemini Pro API for intelligent analysis and SerpAPI for gathering competitive intelligence.

## Features ✨

- **AI-Powered Analysis**: Utilizes Google's Gemini Pro API for intelligent market research analysis
- **Real-time Competitor Research**: Integrates with SerpAPI to fetch current market data and competitor information
- **Comprehensive Reports**: Generates detailed analysis including:
  - Market Overview
  - Target Audience Analysis
  - Competitor Analysis
  - SWOT Analysis
  - Market Size & Trends
  - Strategic Recommendations
  - Resource Requirements
  - Location Analysis
- **Data Visualization**: Creates professional charts and graphs including:
  - Competitor Market Share Distribution (Pie Chart)
  - Market Size and Growth Trends (Line Chart)
- **PDF Report Generation**: Automatically generates well-formatted PDF reports with all analyses and visualizations

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market-analysis-wizard.git
cd market-analysis-wizard
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

## Required Dependencies 📚

- google.generativeai
- serpapi
- python-dotenv
- reportlab
- matplotlib
- logging

## Usage 💡

1. Run the main script:
```bash
python market_research.py
```

2. Enter your business idea when prompted.

3. The tool will:
   - Generate a detailed market analysis
   - Create a JSON output for programmatic use
   - Generate a comprehensive PDF report with visualizations
   - Save all outputs in the project directory

## Output Files 📄

- **PDF Report**: `market_analysis_wizard_[business_idea]_[date].pdf`
- **Log File**: `market_research.log`
- **JSON Data**: Available in console output and can be captured programmatically

## Project Structure 🏗️

```
market-analysis-wizard/
├── market_research.py      # Main application file
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
└── market_research.log    # Application logs
```

## Code Features 🔍

- Object-oriented design with clear separation of concerns
- Comprehensive error handling and logging
- Type hints and data validation
- Clean, well-documented code following PEP 8 guidelines
- Custom exception handling for better error management
- Modular design for easy maintenance and updates

## Error Handling 🚨

The tool includes robust error handling for:
- API connection issues
- JSON parsing errors
- Data validation
- PDF generation problems
- Chart creation errors

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Google Gemini Pro API for AI analysis
- SerpAPI for real-time market data
- ReportLab for PDF generation
- Matplotlib for data visualization

## Contact 📧

Your Name - [@aman-24052001]

Project Link: [https://github.com/aman-24052001/market-analysis-wizard](https://github.com/aman-24052001/market-analysis-wizard)
