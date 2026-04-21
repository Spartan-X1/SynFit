from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
import time
import random
import json

class InfoedgeMasterPipeline:
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        self.output_file = "fresh_naukri_jds.jsonl"

    def discover_urls(self, search_url, pages_to_scrape=3):
        print(f"[*] Starting Discovery Engine on: {search_url}")
        job_urls = set() # Use a set to automatically prevent duplicates

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(user_agent=self.user_agent, viewport={'width': 1920, 'height': 1080})
            page = context.new_page()
            Stealth().apply_stealth_sync(page)

            for i in range(1, pages_to_scrape + 1):
                # Naukri handles pagination by appending "-2", "-3" to the URL
                current_url = f"{search_url}-{i}" if i > 1 else search_url
                print(f"[*] Scanning Search Page {i}: {current_url}")
                
                try:
                    page.goto(current_url, timeout=30000, wait_until="networkidle")
                    
                    # Scroll down slowly to trigger React lazy-loading
                    for _ in range(5):
                        page.mouse.wheel(0, 1000)
                        time.sleep(1)

                    html = page.content()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find all links on the page
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        # Naukri job URLs always contain "job-listings"
                        if "job-listings" in href:
                            job_urls.add(href)

                    print(f"[+] Found {len(job_urls)} unique URLs so far...")
                    time.sleep(random.uniform(4.0, 8.0)) # Sleep between search pages

                except Exception as e:
                    print(f"[!] WAF Block or Timeout on search page {i}: {e}")
                    break

            browser.close()
        
        return list(job_urls)

    def run_extraction_batch(self, url_list):
        print(f"\n[*] Switching to Extraction Engine. Processing {len(url_list)} URLs...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False) 
            context = browser.new_context(user_agent=self.user_agent, viewport={'width': 1920, 'height': 1080})
            page = context.new_page()
            Stealth().apply_stealth_sync(page)
            
            success_count = 0

            for index, url in enumerate(url_list):
                print(f"\n[{index + 1}/{len(url_list)}] Extracting: {url[:60]}...")
                
                try:
                    page.goto(url, timeout=30000, wait_until="networkidle")
                    html = page.content()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    title_elem = soup.find('h1')
                    title = title_elem.text.strip() if title_elem else "Unknown Title"
                    
                    desc_elem = soup.find('div', class_=lambda c: c and 'job-desc' in c.lower())
                    if not desc_elem:
                        desc_elem = soup.find('section', class_=lambda c: c and 'job-desc' in c.lower())
                    
                    raw_text = desc_elem.text.strip() if desc_elem else None

                    if raw_text:
                        clean_text = ' '.join(raw_text.split())
                        record = {
                            "source_url": url,
                            "job_title": title,
                            "raw_jd_text": clean_text
                        }
                        
                        with open(self.output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(record) + '\n')
                            
                        print(f"[+] Saved: {title}")
                        success_count += 1
                    else:
                        print("[-] Failed: No description found.")

                except Exception as e:
                    print(f"[!] Error: {e}")
                
                # Human reading speed delay
                time.sleep(random.uniform(8.0, 18.0))

            print(f"\n[*] Pipeline Complete! Successfully extracted {success_count}/{len(url_list)} JDs.")
            browser.close()

if __name__ == "__main__":
    pipeline = InfoedgeMasterPipeline()
    
    # The Tech Matrix: 20 distinct roles to ensure model generalization
    search_matrix = [
        "https://www.naukri.com/machine-learning-engineer-jobs-in-bengaluru",
        "https://www.naukri.com/frontend-developer-react-jobs-in-bengaluru",
        "https://www.naukri.com/backend-developer-python-jobs-in-bengaluru",
        "https://www.naukri.com/java-spring-boot-developer-jobs-in-bengaluru",
        "https://www.naukri.com/c++developer-jobs-in-bengaluru",
        "https://www.naukri.com/devops-engineer-aws-jobs-in-bengaluru",
        "https://www.naukri.com/site-reliability-engineer-jobs-in-bengaluru",
        "https://www.naukri.com/data-engineer-spark-jobs-in-bengaluru",
        "https://www.naukri.com/full-stack-developer-mern-jobs-in-bengaluru",
        "https://www.naukri.com/android-developer-kotlin-jobs-in-bengaluru",
        "https://www.naukri.com/ios-developer-swift-jobs-in-bengaluru",
        "https://www.naukri.com/product-manager-tech-jobs-in-bengaluru",
        "https://www.naukri.com/qa-automation-engineer-jobs-in-bengaluru",
        "https://www.naukri.com/cyber-security-analyst-jobs-in-bengaluru",
        "https://www.naukri.com/cloud-architect-jobs-in-bengaluru",
        "https://www.naukri.com/blockchain-developer-jobs-in-bengaluru",
        "https://www.naukri.com/ui-ux-designer-jobs-in-bengaluru",
        "https://www.naukri.com/engineering-manager-jobs-in-bengaluru",
        "https://www.naukri.com/database-administrator-jobs-in-bengaluru"
    ]
    
    all_target_urls = []
    
    # 1. Run Discovery across all 20 roles (Scrape 2 pages each = ~40 URLs per role)
    for role_url in search_matrix:
        urls = pipeline.discover_urls(role_url, pages_to_scrape=2)
        all_target_urls.extend(urls)
        # Sleep heavily between role searches so Cloudflare doesn't trip
        time.sleep(random.uniform(15.0, 30.0)) 
    
    print(f"\n[*] GRAND TOTAL URLS FOUND: {len(all_target_urls)}")
    
    # 2. Run the Extraction Engine on the massive list
    if all_target_urls:
        pipeline.run_extraction_batch(all_target_urls)