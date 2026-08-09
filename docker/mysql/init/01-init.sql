-- PaperRAG MySQL 初始化
-- 创建 jpkb 表（BM25 FAQ 知识库，可选）
CREATE TABLE IF NOT EXISTS jpkb (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(20),
    question VARCHAR(1000),
    answer VARCHAR(1000)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建 conversations 表（对话历史）
CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    INDEX idx_session_id (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
