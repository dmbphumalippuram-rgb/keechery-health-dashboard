name: Keep Keechery Dashboard Awake

on:
  schedule:
    # Runs every 8 hours (00:00, 08:00, 16:00 UTC) so Streamlit's 12h sleep timer never triggers
    - cron: '0 */8 * * *'
  workflow_dispatch:

jobs:
  wake-up:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Playwright & Chromium
        run: |
          pip install playwright
          playwright install chromium

      - name: Visit Dashboard & Click Wake Button If Sleeping
        run: |
          python - << 'EOF'
          from playwright.sync_api import sync_playwright

          URLS = [
              "https://keechery-health-dashboard.streamlit.app/",
          ]

          with sync_playwright() as p:
              browser = p.chromium.launch(headless=True)
              page = browser.new_page()
              for url in URLS:
                  print(f"Checking {url}...")
                  page.goto(url, wait_until="networkidle", timeout=60000)
                  page.wait_for_timeout(4000)

                  wake_btn = page.get_by_role("button", name="Yes, get this app back up!")
                  if wake_btn.is_visible():
                      print("App was sleeping — clicking wake-up button...")
                      wake_btn.click()
                      page.wait_for_timeout(20000)
                      print("Wake-up triggered successfully.")
                  else:
                      print("App is already awake and running.")
              browser.close()
          EOF
