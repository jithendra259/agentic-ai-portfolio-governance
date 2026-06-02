import psycopg
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

regions = [
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "ap-east-1", "ap-south-1", "ap-south-2", "ap-northeast-1",
    "ap-northeast-2", "ap-northeast-3", "ap-southeast-1", "ap-southeast-2",
    "ap-southeast-3", "ca-central-1", "eu-central-1", "eu-central-2",
    "eu-west-1", "eu-west-2", "eu-west-3", "eu-north-1",
    "eu-south-1", "sa-east-1", "me-central-1", "me-south-1", "af-south-1"
]

project_ref = "pljsqwdlfpnkupiozgag"
password = "9704400336@Kjs"

def try_region(region):
    import urllib.parse
    host = f"aws-0-{region}.pooler.supabase.com"
    encoded_password = urllib.parse.quote(password)
    conn_str = f"postgresql://postgres.{project_ref}:{encoded_password}@{host}:6543/postgres?connect_timeout=3"
    try:
        conn = psycopg.connect(conn_str)
        conn.close()
        return region, True, None
    except psycopg.OperationalError as e:
        msg = str(e)
        if "tenant/user" in msg or "Tenant or user not found" in msg or "ENOTFOUND" in msg or "getaddrinfo failed" in msg:
            # Not in this region or host doesn't exist
            return region, False, "Tenant not found or DNS failed"
        # Found the region, but something else went wrong (e.g. auth failed, password mismatch)
        return region, True, msg
    except Exception as e:
        return region, False, str(e)

def scan():
    logger.info("Scanning Supabase regions to find project host...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(try_region, r): r for r in regions}
        for future in as_completed(futures):
            region, found, error = future.result()
            if found:
                if error:
                    logger.info(f"===> FOUND REGION: {region} (Region is correct, but got error: {error})")
                else:
                    logger.info(f"===> SUCCESS! CONNECTED TO REGION: {region}")
                sys.exit(0)
            else:
                logger.info(f"Region {region}: {error}")
    logger.error("Scan completed. Project was not found in any Supabase region.")

if __name__ == "__main__":
    scan()
