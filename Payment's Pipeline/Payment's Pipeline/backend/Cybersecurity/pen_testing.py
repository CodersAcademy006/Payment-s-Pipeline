"""
Penetration Testing Module for Payment Pipeline System
Author: AI Assistant
Version: 1.2.0
"""

import asyncio
import json
import logging
import os
import re
import ssl
from datetime import datetime
import socket
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
import colorama
from aiohttp import ClientSession, TCPConnector
from colorama import Fore, Style
from dotenv import load_dotenv

# Initialize colorama for colored console output
colorama.init(autoreset=True)

# Load environment variables
load_dotenv()

class PenTestTool:
    """
    Advanced Penetration Testing Tool for Payment Systems
    """
    
    def __init__(self, target_url: str):
        self.target_url = target_url
        self.session: Optional[ClientSession] = None
        self.report: Dict = {
            "meta": {
                "target": target_url,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.2.0"
            },
            "vulnerabilities": []
        }
        self._setup_logging()
        self._prepare_directories()

    def _setup_logging(self) -> None:
        """Configure logging system"""
        self.logger = logging.getLogger("pentest_tool")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        fh = logging.FileHandler("pentest.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _prepare_directories(self) -> None:
        """Create required directories"""
        os.makedirs("reports", exist_ok=True)
        os.makedirs("payloads", exist_ok=True)

    async def __aenter__(self):
        """Async context manager entry"""
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
        
        connector = TCPConnector(
            ssl=ssl_context,
            limit_per_host=10,
            force_close=True
        )
        
        self.session = ClientSession(
            connector=connector,
            headers={
                "User-Agent": "SecurePenTestBot/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _send_request(self, method: str, endpoint: str, **kwargs) -> Tuple[int, dict]:
        """Safe request handler with retry logic"""
        url = urljoin(self.target_url, endpoint)
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    content = await response.text()
                    try:
                        json_response = await response.json()
                    except json.JSONDecodeError:
                        json_response = None
                    
                    return response.status, {
                        "content": content,
                        "json": json_response,
                        "headers": dict(response.headers)
                    }
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                attempts += 1
                self.logger.warning(f"Attempt {attempts} failed: {str(e)}")
                await asyncio.sleep(2 ** attempts)
        
        self.logger.error(f"Failed to connect to {url} after {max_attempts} attempts")
        return 0, {}

    async def test_sql_injection(self) -> None:
        """Test for SQL injection vulnerabilities"""
        test_endpoints = [
            ("/api/v1/payments", "POST"),
            ("/api/v1/users", "GET")
        ]
        
        payloads = [
            "' OR '1'='1'--",
            "' UNION SELECT NULL, user, password FROM users--"
        ]
        
        for endpoint, method in test_endpoints:
            for payload in payloads:
                data = json.dumps({"query": payload}) if method == "POST" else None
                params = {"id": payload} if method == "GET" else None
                
                status, response = await self._send_request(
                    method=method,
                    endpoint=endpoint,
                    params=params,
                    data=data
                )
                
                if any(error in response.get("content", "").lower() for error in ["sql", "syntax"]):
                    self._add_vulnerability(
                        "SQL Injection",
                        f"Endpoint {endpoint} vulnerable to SQLi",
                        "Critical",
                        {"payload": payload, "response_snippet": response.get("content")[:200]}
                    )

    async def test_xss(self) -> None:
        """Test for Cross-Site Scripting vulnerabilities"""
        test_endpoints = [
            ("/api/v1/search", "GET"),
            ("/api/v1/feedback", "POST")
        ]
        
        payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert(1)"
        ]
        
        for endpoint, method in test_endpoints:
            for payload in payloads:
                data = json.dumps({"input": payload}) if method == "POST" else None
                params = {"q": payload} if method == "GET" else None
                
                status, response = await self._send_request(
                    method=method,
                    endpoint=endpoint,
                    params=params,
                    data=data
                )
                
                if payload in response.get("content", ""):
                    self._add_vulnerability(
                        "Cross-Site Scripting (XSS)",
                        f"Endpoint {endpoint} vulnerable to XSS",
                        "High",
                        {"payload": payload, "response_snippet": response.get("content")[:200]}
                    )

    async def check_pci_compliance(self) -> None:
        """Verify PCI DSS compliance requirements"""
        # Check TLS configuration
        try:
            ssl_context = ssl.create_default_context()
            with socket.create_connection((self.target_url, 443)) as sock:
                with ssl_context.wrap_socket(sock, server_hostname=self.target_url) as ssock:
                    protocol = ssock.version()
                    cipher = ssock.cipher()
                    
                    if "TLSv1.2" not in protocol and "TLSv1.3" not in protocol:
                        self._add_vulnerability(
                            "PCI Compliance",
                            "Insecure TLS version detected",
                            "Critical",
                            {"detected_protocol": protocol}
                        )
        except Exception as e:
            self.logger.error(f"SSL check failed: {str(e)}")

        # Check for sensitive data exposure
        _, payment_response = await self._send_request("GET", "/api/v1/payments/123")
        if "card_number" in payment_response.get("content", ""):
            self._add_vulnerability(
                "PCI Compliance",
                "Sensitive payment data exposed",
                "Critical",
                {"response_snippet": payment_response.get("content")[:200]}
            )

    async def test_auth_bypass(self) -> None:
        """Test for authentication bypass vulnerabilities"""
        endpoints_requiring_auth = [
            "/api/v1/admin/users",
            "/api/v1/payment-config"
        ]
        
        for endpoint in endpoints_requiring_auth:
            status, response = await self._send_request("GET", endpoint)
            if status == 200:
                self._add_vulnerability(
                    "Authentication Bypass",
                    f"Unauthorized access to {endpoint}",
                    "Critical",
                    {"status_code": status}
                )

    async def test_insecure_direct_object_reference(self) -> None:
        """Test for Insecure Direct Object Reference (IDOR)"""
        test_user_id = "1001"
        _, response = await self._send_request(
            "GET",
            f"/api/v1/users/{test_user_id}/payments"
        )
        if response.get("status") == 200:
            self._add_vulnerability(
                "Insecure Direct Object Reference",
                "Access to unauthorized resources possible",
                "High",
                {"tested_endpoint": f"/users/{test_user_id}/payments"}
            )

    async def test_api_security(self) -> None:
        """Test general API security best practices"""
        # Check for proper CORS settings
        _, response = await self._send_request("OPTIONS", "/api/v1/payments")
        if response.get("headers", {}).get("Access-Control-Allow-Origin") == "*":
            self._add_vulnerability(
                "API Security",
                "Insecure CORS configuration detected",
                "Medium",
                {"headers": response.get("headers")}
            )

        # Check for missing rate limiting
        for _ in range(20):
            await self._send_request("GET", "/api/v1/payments")
        
        status, response = await self._send_request("GET", "/api/v1/payments")
        if status != 429:
            self._add_vulnerability(
                "API Security",
                "Missing rate limiting protection",
                "Medium",
                {"requests_sent": 20, "final_status": status}
            )

    def _add_vulnerability(self, title: str, description: str, severity: str, evidence: dict) -> None:
        """Add vulnerability to report"""
        vuln = {
            "title": title,
            "description": description,
            "severity": severity,
            "evidence": evidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.report["vulnerabilities"].append(vuln)
        
        # Print to console with color coding
        color = {
            "Critical": Fore.RED,
            "High": Fore.YELLOW,
            "Medium": Fore.CYAN,
            "Low": Fore.WHITE
        }.get(severity, Fore.WHITE)
        
        print(f"{color}[{severity}] {title}{Style.RESET_ALL}")
        print(f"Description: {description}\n")

    async def run_all_tests(self) -> None:
        """Execute complete test suite"""
        test_methods = [
            self.test_sql_injection,
            self.test_xss,
            self.check_pci_compliance,
            self.test_auth_bypass,
            self.test_insecure_direct_object_reference,
            self.test_api_security
        ]
        
        tasks = [asyncio.create_task(test()) for test in test_methods]
        await asyncio.gather(*tasks)
        
        self._generate_report()
        self._print_summary()

    def _generate_report(self) -> None:
        """Generate final report"""
        filename = f"reports/report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.report, f, indent=2)
            
        self.logger.info(f"Report generated: {filename}")

    def _print_summary(self) -> None:
        """Print summary of findings"""
        print(Fore.GREEN + "\n=== Penetration Test Summary ===")
        print(f"Target: {self.target_url}")
        print(f"Total vulnerabilities found: {len(self.report['vulnerabilities'])}")
        
        severity_counts = {}
        for vuln in self.report["vulnerabilities"]:
            severity = vuln["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        for severity, count in severity_counts.items():
            color = {
                "Critical": Fore.RED,
                "High": Fore.YELLOW,
                "Medium": Fore.CYAN,
                "Low": Fore.WHITE
            }.get(severity, Fore.WHITE)
            print(f"{color}{severity}: {count}{Style.RESET_ALL}")
            
        print(Fore.GREEN + "=== End of Report ===")

if __name__ == "__main__":
    async def main():
        target_url = input("Enter target URL (e.g., https://api.payments.com): ").strip()
        if not target_url.startswith("http"):
            target_url = f"https://{target_url}"
            
        async with PenTestTool(target_url) as tester:
            await tester.run_all_tests()

    asyncio.run(main())