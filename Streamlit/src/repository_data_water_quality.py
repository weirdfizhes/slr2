# src/repository_data_water_quality.py

from typing import List, Dict
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data_water_quality(db_conn, lon1: float, lon2: float, lat1: float, lat2: float) -> List[Dict]:
    """
    Get water quality data within specified coordinate bounds.
    Uses spatial index for efficient querying.
    
    Args:
        db_conn: Database connection
        lon1, lon2: Longitude bounds
        lat1, lat2: Latitude bounds
    
    Returns:
        List of dictionaries containing water quality measurements
    """
    try:
        # Ensure coordinates are in correct order
        lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
        lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)

        # First, check the query plan
        explain_query = """
        EXPLAIN ANALYZE
        SELECT longitude, latitude, temp, salinity, "do", ph, tss, bathy
        FROM measurements
        WHERE longitude BETWEEN %s AND %s
          AND latitude BETWEEN %s AND %s
          AND bathy < 0
        ORDER BY longitude, latitude;
        """
        
        # Actual data query without EXPLAIN ANALYZE
        data_query = """
        SELECT longitude, latitude, temp, salinity, "do", ph, tss, bathy
        FROM measurements
        WHERE longitude BETWEEN %s AND %s
          AND latitude BETWEEN %s AND %s
          AND bathy < 0
        ORDER BY longitude, latitude;
        """

        with db_conn.cursor() as cur:
            # First, analyze the query plan
            cur.execute(explain_query, (lon_min, lon_max, lat_min, lat_max))
            plan = cur.fetchall()
            logger.debug("Query plan:")
            for row in plan:
                logger.debug(row[0])

            # Then execute the actual query
            cur.execute(data_query, (lon_min, lon_max, lat_min, lat_max))
            rows = cur.fetchall()
            
            # Convert rows to list of dictionaries
            result = [
                {
                    "longitude": row[0],
                    "latitude": row[1],
                    "temp": row[2],
                    "salinity": row[3],
                    "do": row[4],
                    "ph": row[5],
                    "tss": row[6],
                    "bathy": row[7]
                }
                for row in rows
            ]
            
            logger.info(f"Retrieved {len(result)} measurements from database")
            return result

    except Exception as e:
        logger.error(f"Error retrieving water quality data: {e}")
        raise