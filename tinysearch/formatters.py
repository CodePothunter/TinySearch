"""
Response formatting utilities for TinySearch
"""
from typing import Dict, List, Any, Optional, Union
import json
import re
import html
from datetime import datetime
from pathlib import Path


class ResponseFormatter:
    """
    Base class for response formatters
    """
    
    def format_response(self, results: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the response from the query engine
        
        Args:
            results: List of results from the query engine
            **kwargs: Additional formatting options
            
        Returns:
            Formatted response
        """
        raise NotImplementedError("Subclasses must implement this method")


class PlainTextFormatter(ResponseFormatter):
    """
    Format results as plain text
    """
    
    def __init__(
        self,
        include_metadata: bool = True,
        include_scores: bool = True,
        separator: str = "\n\n---\n\n"
    ):
        """
        Initialize the plain text formatter
        
        Args:
            include_metadata: Whether to include metadata in the output
            include_scores: Whether to include similarity scores in the output
            separator: Separator between results
        """
        self.include_metadata = include_metadata
        self.include_scores = include_scores
        self.separator = separator
    
    def format_response(self, results: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the response as plain text
        
        Args:
            results: List of results from the query engine
            **kwargs: Additional formatting options
            
        Returns:
            Formatted plain text response
        """
        if not results:
            return "No results found."
        
        formatted_results = []
        
        for i, result in enumerate(results):
            parts = []
            
            # Add the text content
            text = result.get("text", "")
            parts.append(text)
            
            # Add the metadata
            if self.include_metadata and "metadata" in result and result["metadata"]:
                meta_str = self._format_metadata(result["metadata"])
                if meta_str:
                    parts.append(f"Metadata: {meta_str}")
            
            # Add the similarity score
            if self.include_scores and "score" in result:
                score = result["score"]
                parts.append(f"Similarity: {score:.4f}")
                
            # Join all parts and add to formatted results
            formatted_results.append("\n".join(parts))
            
        return self.separator.join(formatted_results)
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata as a string
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Formatted metadata string
        """
        if not metadata:
            return ""
        
        # Handle special case for source
        if "source" in metadata:
            source = metadata["source"]
            if isinstance(source, str):
                try:
                    path = Path(source)
                    return f"Source: {path.name}"
                except:
                    return f"Source: {source}"
        
        # For other metadata, format as key-value pairs
        pairs = []
        for key, value in metadata.items():
            if isinstance(value, dict):
                continue  # Skip nested dictionaries for simplicity
            pairs.append(f"{key}: {value}")
        
        return ", ".join(pairs)


class MarkdownFormatter(ResponseFormatter):
    """
    Format results as Markdown
    """
    
    def __init__(
        self,
        include_metadata: bool = True,
        include_scores: bool = True,
        link_sources: bool = True
    ):
        """
        Initialize the Markdown formatter
        
        Args:
            include_metadata: Whether to include metadata in the output
            include_scores: Whether to include similarity scores in the output
            link_sources: Whether to convert source paths to Markdown links
        """
        self.include_metadata = include_metadata
        self.include_scores = include_scores
        self.link_sources = link_sources
    
    def format_response(self, results: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the response as Markdown
        
        Args:
            results: List of results from the query engine
            **kwargs: Additional formatting options
            
        Returns:
            Formatted Markdown response
        """
        if not results:
            return "No results found."
        
        lines = ["# Search Results", ""]
        
        for i, result in enumerate(results):
            # Result header
            lines.append(f"## Result {i+1}")
            lines.append("")
            
            # Add the similarity score
            if self.include_scores and "score" in result:
                score = result["score"]
                lines.append(f"*Similarity: {score:.4f}*")
                lines.append("")
            
            # Add the text content
            text = result.get("text", "")
            lines.append(f"```\n{text}\n```")
            lines.append("")
            
            # Add the metadata
            if self.include_metadata and "metadata" in result and result["metadata"]:
                lines.append("**Metadata:**")
                lines.append("")
                meta_lines = self._format_metadata(result["metadata"])
                lines.extend(meta_lines)
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Format metadata as Markdown lines
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            List of Markdown-formatted lines
        """
        if not metadata:
            return []
        
        lines = []
        
        # Handle special case for source
        if "source" in metadata:
            source = metadata["source"]
            if isinstance(source, str):
                try:
                    path = Path(source)
                    if self.link_sources:
                        lines.append(f"- **Source**: [{path.name}]({source})")
                    else:
                        lines.append(f"- **Source**: {path.name}")
                except:
                    lines.append(f"- **Source**: {source}")
        
        # For other metadata, format as bullet points
        for key, value in metadata.items():
            if key == "source":
                continue  # Already handled
            if isinstance(value, dict):
                continue  # Skip nested dictionaries for simplicity
            lines.append(f"- **{key}**: {value}")
        
        return lines


class JSONFormatter(ResponseFormatter):
    """
    Format results as JSON
    """
    
    def __init__(
        self,
        pretty: bool = False,
        include_timestamp: bool = True
    ):
        """
        Initialize the JSON formatter
        
        Args:
            pretty: Whether to format the JSON with indentation
            include_timestamp: Whether to include a timestamp in the output
        """
        self.pretty = pretty
        self.include_timestamp = include_timestamp
    
    def format_response(self, results: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the response as JSON
        
        Args:
            results: List of results from the query engine
            **kwargs: Additional formatting options
            
        Returns:
            Formatted JSON response
        """
        response = {
            "results": results,
            "count": len(results)
        }
        
        if self.include_timestamp:
            response["timestamp"] = datetime.now().isoformat()
        
        if self.pretty:
            return json.dumps(response, indent=2)
        else:
            return json.dumps(response)


class HTMLFormatter(ResponseFormatter):
    """
    Format results as HTML
    """
    
    def __init__(
        self,
        include_metadata: bool = True,
        include_scores: bool = True,
        include_css: bool = True
    ):
        """
        Initialize the HTML formatter
        
        Args:
            include_metadata: Whether to include metadata in the output
            include_scores: Whether to include similarity scores in the output
            include_css: Whether to include CSS styling in the output
        """
        self.include_metadata = include_metadata
        self.include_scores = include_scores
        self.include_css = include_css
    
    def format_response(self, results: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the response as HTML
        
        Args:
            results: List of results from the query engine
            **kwargs: Additional formatting options
            
        Returns:
            Formatted HTML response
        """
        if not results:
            return "<div class='tinysearch-results'><p>No results found.</p></div>"
        
        html_parts = []
        
        # Add CSS if requested
        if self.include_css:
            html_parts.append(self._get_css())
        
        # Start the results container
        html_parts.append("<div class='tinysearch-results'>")
        
        for i, result in enumerate(results):
            # Start result div
            html_parts.append(f"<div class='tinysearch-result' data-result-index='{i}'>")
            
            # Add the similarity score
            if self.include_scores and "score" in result:
                score = result["score"]
                html_parts.append(f"<div class='tinysearch-score'>Similarity: {score:.4f}</div>")
            
            # Add the text content
            text = html.escape(result.get("text", "")).replace("\n", "<br>")
            html_parts.append(f"<div class='tinysearch-text'>{text}</div>")
            
            # Add the metadata
            if self.include_metadata and "metadata" in result and result["metadata"]:
                html_parts.append("<div class='tinysearch-metadata'>")
                meta_html = self._format_metadata_html(result["metadata"])
                html_parts.append(meta_html)
                html_parts.append("</div>")
            
            # End result div
            html_parts.append("</div>")
        
        # End the results container
        html_parts.append("</div>")
        
        return "\n".join(html_parts)
    
    def _format_metadata_html(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata as HTML
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            HTML-formatted metadata
        """
        if not metadata:
            return ""
        
        html_parts = ["<dl>"]
        
        # Handle special case for source
        if "source" in metadata:
            source = metadata["source"]
            if isinstance(source, str):
                try:
                    path = Path(source)
                    html_parts.append("<dt>Source</dt>")
                    html_parts.append(f"<dd>{html.escape(path.name)}</dd>")
                except:
                    html_parts.append("<dt>Source</dt>")
                    html_parts.append(f"<dd>{html.escape(str(source))}</dd>")
        
        # For other metadata, format as definition list
        for key, value in metadata.items():
            if key == "source":
                continue  # Already handled
            if isinstance(value, dict):
                continue  # Skip nested dictionaries for simplicity
            html_parts.append(f"<dt>{html.escape(str(key))}</dt>")
            html_parts.append(f"<dd>{html.escape(str(value))}</dd>")
        
        html_parts.append("</dl>")
        return "\n".join(html_parts)
    
    def _get_css(self) -> str:
        """
        Get CSS styles for the HTML output
        
        Returns:
            CSS styles as a string
        """
        return """
        <style>
        .tinysearch-results {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
        }
        .tinysearch-result {
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .tinysearch-score {
            font-size: 0.8em;
            color: #666;
            margin-bottom: 5px;
        }
        .tinysearch-text {
            margin-bottom: 10px;
            white-space: pre-wrap;
        }
        .tinysearch-metadata {
            font-size: 0.9em;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }
        .tinysearch-metadata dl {
            margin: 0;
            display: grid;
            grid-template-columns: max-content auto;
            gap: 5px 10px;
        }
        .tinysearch-metadata dt {
            font-weight: bold;
            color: #555;
        }
        .tinysearch-metadata dd {
            margin: 0;
        }
        </style>
        """


def get_formatter(format_type: str, **kwargs) -> ResponseFormatter:
    """
    Factory function to get a formatter by type
    
    Args:
        format_type: Type of formatter ('text', 'markdown', 'json', 'html')
        **kwargs: Additional options for the formatter
        
    Returns:
        Appropriate formatter instance
    """
    format_type = format_type.lower()
    
    if format_type == "text":
        return PlainTextFormatter(**kwargs)
    elif format_type == "markdown":
        return MarkdownFormatter(**kwargs)
    elif format_type == "json":
        return JSONFormatter(**kwargs)
    elif format_type == "html":
        return HTMLFormatter(**kwargs)
    else:
        raise ValueError(f"Unknown format type: {format_type}") 