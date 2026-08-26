-- 1. Monthly Delay Pattern
SELECT posting_month, COUNT(*) as total_invoices, AVG(is_open)*100 as delayed_pct 
FROM invoices GROUP BY posting_month ORDER BY delayed_pct DESC;

-- 2. Top 10 risky customers
SELECT cust_number, COUNT(*) as total_invoices, AVG(is_open)*100 as risk_score
FROM invoices GROUP BY cust_number HAVING COUNT(*) > 5
ORDER BY risk_score DESC LIMIT 10;

-- 3. Risk by amount bucket
SELECT 
 CASE WHEN total_open_amount < 50000 THEN 'Low'
      WHEN total_open_amount < 200000 THEN 'Medium'
      ELSE 'High' END as bucket,
 AVG(is_open)*100 as delay_rate
FROM invoices GROUP BY bucket;

-- 4. Quarter-End Effect
SELECT CASE WHEN posting_month IN (3,6,9,12) THEN 'Quarter-End' ELSE 'Normal' END as period_type,
AVG(is_open)*100 as delay_rate FROM invoices GROUP BY period_type;