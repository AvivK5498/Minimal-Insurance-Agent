# Run with: uv run mcp_server.py
"""
MCP Server for Insurance Claim Policy Validation Tools.
Provides tools for validating policy limits and calculating date differences.
"""

from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Insurance Policy Tools")


@mcp.tool()
def validate_policy_limit(claimed_amount: float, policy_limit: float) -> str:
    """
    Validates if a claimed amount is within the policy coverage limits.
    
    Args:
        claimed_amount: The total amount being claimed in dollars
        policy_limit: The maximum coverage limit from the policy in dollars
    
    Returns:
        A structured analysis string indicating if the claim is within limits
    """
    if claimed_amount < 0:
        return "ERROR: Claimed amount cannot be negative."
    
    if policy_limit <= 0:
        return "ERROR: Policy limit must be a positive value."
    
    percentage_used = (claimed_amount / policy_limit) * 100
    remaining = policy_limit - claimed_amount
    
    if claimed_amount <= policy_limit:
        risk_level = "Low" if percentage_used < 50 else "Medium" if percentage_used < 80 else "High"
        return (
            f"VALIDATION RESULT: Claim is WITHIN policy limits.\n"
            f"- Claimed Amount: ${claimed_amount:,.2f}\n"
            f"- Policy Limit: ${policy_limit:,.2f}\n"
            f"- Coverage Used: {percentage_used:.1f}%\n"
            f"- Remaining Coverage: ${remaining:,.2f}\n"
            f"- Risk Level: {risk_level}"
        )
    else:
        excess = claimed_amount - policy_limit
        return (
            f"VALIDATION RESULT: Claim EXCEEDS policy limits.\n"
            f"- Claimed Amount: ${claimed_amount:,.2f}\n"
            f"- Policy Limit: ${policy_limit:,.2f}\n"
            f"- Amount Over Limit: ${excess:,.2f}\n"
            f"- Risk Level: HIGH\n"
            f"- Action Required: Review for partial coverage or denial."
        )


@mcp.tool()
def calculate_date_difference(date1: str, date2: str) -> str:
    """
    Calculates the difference in days between two dates.
    Useful for timeline analysis in insurance claims.
    
    Args:
        date1: First date in format 'YYYY-MM-DD' or 'Month DD, YYYY'
        date2: Second date in format 'YYYY-MM-DD' or 'Month DD, YYYY'
    
    Returns:
        A string describing the time difference between the dates
    """
    formats = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"]
    
    def parse_date(date_str: str) -> datetime:
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse date: {date_str}")
    
    try:
        d1 = parse_date(date1)
        d2 = parse_date(date2)
        
        diff = abs((d2 - d1).days)
        earlier = d1 if d1 < d2 else d2
        later = d2 if d1 < d2 else d1
        
        return (
            f"DATE ANALYSIS:\n"
            f"- Earlier Date: {earlier.strftime('%B %d, %Y')}\n"
            f"- Later Date: {later.strftime('%B %d, %Y')}\n"
            f"- Difference: {diff} days\n"
            f"- Weeks: {diff // 7} weeks and {diff % 7} days"
        )
    except ValueError as e:
        return f"ERROR: {str(e)}. Please use format 'YYYY-MM-DD' or 'Month DD, YYYY'."


if __name__ == "__main__":
    mcp.run()



