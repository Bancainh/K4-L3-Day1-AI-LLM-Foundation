# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Với temperature 0.0, phản hồi mang tính xác định cao, chuẩn mực và gần như không đổi giữa các lần gọi. Khi tăng lên 0.5 và 1.0, mô hình sáng tạo hơn, cung cấp các sự thật đa dạng và cách diễn đạt tự nhiên, phong phú hơn. Ở mức 1.5, mô hình bị ép phải chọn các token có xác suất thấp, dẫn đến hiện tượng ảo giác (hallucination), lặp từ hoặc câu văn mất tính logic mạch lạc.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ đặt temperature ở mức thấp (từ 0.0 đến 0.2). Chatbot hỗ trợ khách hàng cần cung cấp thông tin chính xác, nhất quán và tuân thủ chặt chẽ tài liệu nghiệp vụ của công ty, tuyệt đối tránh việc "sáng tạo" hay bịa đặt thông tin (ảo giác) làm ảnh hưởng đến trải nghiệm người dùng và uy tín doanh nghiệp.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Dựa trên bảng giá, chi phí đầu ra của GPT-4o đắt hơn GPT-4o-mini khoảng 16,6 lần (0.010 USD so với 0.0006 USD cho mỗi 1000 token). GPT-4o xứng đáng với chi phí khi cần xử lý các tác vụ phức tạp đòi hỏi tư duy logic sâu, lập trình hoặc phân tích ngữ cảnh tinh tế (ví dụ: review hợp đồng pháp lý). Ngược lại, GPT-4o-mini là lựa chọn tối ưu cho các tác vụ đơn giản, lặp đi lặp lại với khối lượng lớn như phân loại văn bản, trích xuất thực thể, hoặc chatbot hỏi đáp FAQ thông thường.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Phản hồi của "giáo viên tiểu học" ngắn gọn, dùng từ ngữ gần gũi, so sánh trực quan (như "cuốn sổ chung của lớp") và có giọng điệu thân thiện. Trong khi đó, phản hồi của "chuyên gia tài chính" dài hơn, cấu trúc phức tạp và sử dụng dày đặc các thuật ngữ chuyên ngành (sổ cái phân tán, mã hóa cryptography, node, smart contract). Điều này chứng minh system prompt đóng vai trò như "đạo diễn", thiết lập và giới hạn chặt chẽ phạm vi từ vựng, mức độ phức tạp cũng như phong cách giao tiếp của mô hình trước khi xử lý yêu cầu thực tế.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Số token đếm bằng thư viện tiktoken thường cao hơn đáng kể (có thể chênh lệch 30% - 50% hoặc hơn) so với công thức ước lượng thô. Tiếng Việt tốn nhiều token hơn tiếng Anh vì các bộ mã hóa (như o200k_base của OpenAI) được huấn luyện tối ưu chủ yếu cho kho ngữ liệu tiếng Anh; do đó, các từ tiếng Việt có dấu, ký tự đặc biệt, hoặc từ ghép thường bị thuật toán xé lẻ thành 2-3 token nhỏ lẻ thay vì gộp chung thành 1 token trọn vẹn như từ vựng tiếng Anh.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming đặc biệt quan trọng trong các giao diện tương tác trực tiếp (như chatbot hoặc AI assistant), giúp giảm thiểu độ trễ cảm nhận (perceived latency) vì người dùng có thể đọc ngay những chữ đầu tiên thay vì phải nhìn màn hình chờ trống không suốt hàng chục giây. Ngược lại, non-streaming lại phù hợp hơn cho các tác vụ xử lý ngầm (background jobs), phân tích dữ liệu hàng loạt (batch processing), hoặc khi hệ thống cần nhận về toàn bộ dữ liệu có cấu trúc hoàn chỉnh (ví dụ: JSON object) để thực hiện các bước logic code tiếp theo mà không cần hiển thị trung gian cho con người.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp giảm tải áp lực lên server đang bị nghẽn bằng cách tự động dãn cách thời gian thử lại ngày càng thưa hơn. Nếu hàng nghìn client cùng retry với một delay cố định 1 giây, chúng sẽ vô tình tạo ra một đợt tấn công DDoS cục bộ (hiện tượng thundering herd) dội vào hệ thống ngay tại cùng một thời điểm khi server vừa kịp phục hồi, khiến server tiếp tục sập ngay lập tức do quá tải đột ngột.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Persona: "Bạn là một trợ lý lập trình chuyên nghiệp. Luôn trả lời bằng mã nguồn tối ưu nhất, không giải thích lan man, chỉ sử dụng tiếng Việt."
Lựa chọn "không giải thích lan man" giúp tiết kiệm triệt để lượng token đầu ra, từ đó giảm chi phí và tăng tốc độ phản hồi. Lựa chọn "chỉ sử dụng tiếng Việt" là chỉ định ngôn ngữ bắt buộc để ép mô hình không tự động chuyển sang tiếng Anh khi gặp các thuật ngữ kỹ thuật phức tạp, đảm bảo tính thống nhất trong trải nghiệm người dùng.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất là history bị cắt cứng chỉ còn 3 lượt (6 tin nhắn), khiến trợ lý mắc bệnh "trí nhớ ngắn hạn" và hoàn toàn quên mất các bối cảnh quan trọng ở phần đầu cuộc hội thoại dài. Để cải thiện, có thể triển khai cơ chế "Tóm tắt ngữ cảnh tự động" (Context Summarization): khi history đạt đến giới hạn, thay vì xóa bỏ hoàn toàn các tin nhắn cũ, hệ thống sẽ gọi một lượt API phụ ngầm để tóm tắt những tin nhắn sắp bị xóa thành một đoạn văn ngắn gọn, sau đó chèn đoạn tóm tắt này vào ngay sau system prompt để giữ lại ý chính mà vẫn tiết kiệm đáng kể số lượng token.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
