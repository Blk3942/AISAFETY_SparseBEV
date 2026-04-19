-- 经 SSH 反向隧道写入「本机 MySQL」时的连通性测试表
-- 用法（在远端执行，把用户/路径按实际修改）：
--   mysql -h 127.0.0.1 -P 13306 -u remote -p < AISAFETY_SparseBEV/docs/mysql_tunnel_test_table.sql

USE safetyai_sparsebev;

CREATE TABLE IF NOT EXISTS tunnel_connectivity_test (
    id         BIGINT AUTO_INCREMENT PRIMARY KEY,
    note       VARCHAR(255) NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO tunnel_connectivity_test (note) VALUES ('tunnel_write_ok');
