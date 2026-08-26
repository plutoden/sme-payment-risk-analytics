-- Production Risk Scoring Logic (Bank Core)
CREATE VIEW sme_risk_score AS
SELECT cust_number, total_open_amount,
CASE 
 WHEN total_open_amount > 100000 THEN 85
 WHEN total_open_amount > 75000 THEN 75
 WHEN total_open_amount > 50000 THEN 55
 ELSE 15
END as risk_score
FROM invoices;