# Parallel BFS – MPI + OpenMP

Thuật toán **Breadth-First Search song song phân tán** sử dụng:
- **MPI** để giao tiếp giữa các máy / tiến trình
- **OpenMP** để song song hóa trong từng tiến trình
- **Direction-Optimizing BFS** (Top-Down + Bottom-Up) theo Beamer et al. (2012)

---

## Cấu trúc project

```
.
├── graph.h / graph.c          # CSR graph, đọc file, sinh đồ thị RMAT & Erdős–Rényi
├── partition.h / partition.c  # Phân vùng đỉnh 1D-Block cho các tiến trình
├── comm.h / comm.c            # Giao tiếp MPI giữa các máy (Alltoallv, Allgatherv)
├── bfs_core.h / bfs_core.c   # Thuật toán BFS Top-Down / Bottom-Up + OpenMP
├── main.c                     # Điều phối, benchmark TEPS, validate, xuất file
└── Makefile
```

---

## Yêu cầu

| Phần mềm | Phiên bản tối thiểu |
|---|---|
| GCC | 9+ |
| OpenMPI | 4.0+ |
| OpenMP | tích hợp trong GCC |

---

## Build

```bash
make            # build bình thường
make clean      # xóa file build
make clean && make   # build lại từ đầu
```

---

## Cách chạy

### Tham số dòng lệnh

```
./bfs --rmat <scale> <edgefactor>   # Sinh đồ thị RMAT: N=2^scale, M=edgefactor×N
./bfs --er   <N> <p>                # Sinh đồ thị Erdős–Rényi: N đỉnh, xác suất cạnh p
./bfs --file <graph.txt>            # Đọc đồ thị từ file

Tùy chọn bổ sung:
  --root     <v>   Đỉnh nguồn BFS (mặc định: 0)
  --out      <f>   Xuất kết quả khoảng cách ra file
  --bench    <k>   Chạy BFS k lần, báo cáo TEPS trung bình
  --validate       So sánh kết quả với BFS tuần tự (chỉ dùng khi N ≤ 2 triệu)
```

### Định dạng file đồ thị

```
N M
u1 v1
u2 v2
...
```
Trong đó `N` = số đỉnh, `M` = số cạnh, các đỉnh đánh số từ `0`.

---

## Chạy trên 1 máy

```bash
# Chạy với 2 tiến trình (phù hợp máy 2 core)
mpirun -np 2 ./bfs --rmat 16 8 --validate

# Nếu muốn dùng nhiều tiến trình hơn số core (ép chạy)
mpirun -np 4 --oversubscribe ./bfs --rmat 18 16 --validate

# Dùng OpenMP: 2 tiến trình × 2 thread = 4 luồng tổng
OMP_NUM_THREADS=2 mpirun -np 2 ./bfs --rmat 18 16 --bench 3

# Chạy từ file, xuất kết quả ra dist.txt
mpirun -np 2 ./bfs --file graph.txt --root 0 --out dist.txt --validate
```

---

## Chạy nhiều máy (MPI cluster)

Thực hiện **tuần tự từng bước** trên tất cả các máy (master + các slave), trừ khi có ghi chú riêng.

---

### Bước 1 – Cập nhật hệ thống (tất cả máy)

```bash
sudo apt-get update
sudo apt-get upgrade
```

> Nhấn `y` nếu được hỏi.

---

### Bước 2 – Kiểm tra địa chỉ IP (tất cả máy)

```bash
sudo apt-get install net-tools
ifconfig
```

Ghi lại địa chỉ IP của từng máy để dùng ở các bước sau.

---

### Bước 3 – Cài SSH server + client (tất cả máy)

```bash
sudo apt-get install openssh-server
sudo apt-get install openssh-client
```

> Nhấn `y` nếu được hỏi.

---

### Bước 4 – Cài OpenMPI + GCC (tất cả máy)

```bash
sudo apt-get install -y libopenmpi-dev openmpi-bin gcc
```

---

### Bước 5 – Tạo thư mục `.ssh` (tất cả máy)

```bash
mkdir ~/.ssh
chmod 700 ~/.ssh
```

---

### Bước 6 – Sinh RSA key (tất cả máy)

```bash
ssh-keygen -t rsa
```

Khi được hỏi:
- **Tên file**: nhập đường dẫn trong `.ssh`, ví dụ:
  - Trên master: `~/.ssh/id_rsa_master`
  - Trên slave1:  `~/.ssh/id_rsa_slave1`
  - Trên slave2:  `~/.ssh/id_rsa_slave2`
- **Passphrase**: nhập mật khẩu cho file RSA (nhớ lại để dùng ở Bước 9)

---

### Bước 7 – Trao đổi public key giữa các máy

Copy file `.pub` từ slave về master, và ngược lại:

```
id_rsa_slave1.pub  (slave1  → master)  vào /home/mpiuser/.ssh/
id_rsa_slave2.pub  (slave2  → master)  vào /home/mpiuser/.ssh/
id_rsa_master.pub  (master  → slave1)  vào /home/mpiuser/.ssh/
id_rsa_master.pub  (master  → slave2)  vào /home/mpiuser/.ssh/
```

Sau khi copy xong, chạy lệnh sau để thêm vào `authorized_keys`:

**Trên master:**
```bash
cat /home/mpiuser/.ssh/id_rsa_slave1.pub >> /home/mpiuser/.ssh/authorized_keys
cat /home/mpiuser/.ssh/id_rsa_slave2.pub >> /home/mpiuser/.ssh/authorized_keys
```

**Trên slave1:**
```bash
cat /home/mpiuser/.ssh/id_rsa_master.pub >> /home/mpiuser/.ssh/authorized_keys
```

**Trên slave2:**
```bash
cat /home/mpiuser/.ssh/id_rsa_master.pub >> /home/mpiuser/.ssh/authorized_keys
```

---

### Bước 8 – Cấu hình SSH (tất cả máy)

Cài gedit để chỉnh sửa file cấu hình:

```bash
sudo apt-get install gedit
```

Mở file cấu hình SSH:

```bash
sudo gedit /etc/ssh/sshd_config
```

Thêm hai dòng sau vào cuối file rồi lưu lại:

```
PubkeyAuthentication yes
RSAAuthentication yes
```

Khởi động lại SSH:

```bash
sudo service ssh restart
```

---

### Bước 9 – Kiểm tra kết nối SSH

**Trên master**, thử SSH sang từng slave:
```bash
ssh mpiuser@<ip_slave1>
ssh mpiuser@<ip_slave2>
```

**Trên slave**, thử SSH sang master:
```bash
ssh mpiuser@<ip_master>
```

> Nhấn `yes` nếu được hỏi lần đầu. Nhập passphrase của RSA key khi được yêu cầu.

---

### Bước 10 – Đồng bộ code lên tất cả máy

Chạy trên **master**:

```bash
rsync -avz ~/parrallel_prj/ mpiuser@<ip_slave1>:~/parrallel_prj/
rsync -avz ~/parrallel_prj/ mpiuser@<ip_slave2>:~/parrallel_prj/
```

---

### Bước 11 – Build trên tất cả máy

SSH vào từng slave và build:

```bash
ssh mpiuser@<ip_slave1> "cd ~/parrallel_prj && make clean && make"
ssh mpiuser@<ip_slave2> "cd ~/parrallel_prj && make clean && make"
```

---

### Bước 12 – Tạo hostfile

Tạo file `hostfile` trên **master**:

```
# hostfile
<ip_master> slots=2    # máy chủ (rank 0)
<ip_slave1> slots=2    # slave 1
<ip_slave2> slots=2    # slave 2
```

`slots=N` là số tiến trình MPI chạy trên máy đó (thường = số CPU core).

Kiểm tra số core của từng máy:
```bash
nproc
```

---

### Bước 13 – Chạy

```bash
# 6 tiến trình tổng (3 máy × 2 tiến trình), 2 thread/tiến trình
OMP_NUM_THREADS=2 mpirun -np 6 --hostfile hostfile ./bfs --rmat 20 16 --bench 5

# Validate kết quả (đồ thị nhỏ)
OMP_NUM_THREADS=2 mpirun -np 4 --hostfile hostfile ./bfs --rmat 16 8 --validate

# Xuất kết quả ra file
OMP_NUM_THREADS=2 mpirun -np 4 --hostfile hostfile ./bfs --file graph.txt --out dist.txt
```

---

## Giải thích output

```
╔══════════════════════════════════════════════╗
║   Parallel BFS – MPI + OpenMP (5-file ver)  ║
╠══════════════════════════════════════════════╣
║  Source      : RMAT                         ║
║  MPI procs   : 4                            ║
║  OMP threads : 2       (total 8    )        ║
║  Vertices N  : 262144                       ║
║  Edges    M  : 4194304                      ║
║  BFS root    : 0                            ║
╚══════════════════════════════════════════════╝

  Level   0 | TD | frontier=1        ← TD = Top-Down
  Level   1 | TD | frontier=12
  Level   2 | BU | frontier=1842     ← BU = Bottom-Up (frontier lớn)
  Level   3 | TD | frontier=203

┌─────────────────── BFS Result ───────────────────┐
│  Visited   : 261980 / 262144  (99.9%)
│  BFS depth : 7 levels
│  Time      : 0.0842 s
│  TEPS      : 4.98e+07              ← Traversed Edges Per Second
│  Procs×Thr : 4 × 2 = 8
└──────────────────────────────────────────────────┘

[VALID] ✓ Correct (N=262144)         ← kết quả đúng so với BFS tuần tự
```

**TEPS** (Traversed Edges Per Second) là chỉ số hiệu năng chuẩn của Graph500. TEPS càng cao càng tốt.

---

## Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `not enough slots` | Số tiến trình > số core | Thêm `--oversubscribe` hoặc giảm `-np` |
| `cannot connect to host` | SSH chưa được cấu hình | Làm lại Bước 9 |
| `Connection refused` | SSH server chưa cài hoặc chưa chạy | Làm lại Bước 3, kiểm tra `sudo service ssh status` |
| `bfs_core.h: No such file` | Thiếu file | Clone lại repo, kiểm tra đủ file |
| `[VALID] ✗ errors` | Kết quả sai | Báo lại để debug |

---

## Tài liệu tham khảo

- Beamer, S., Asanović, K., & Patterson, D. (2012). *Direction-Optimizing Breadth-First Search*. SC '12.
- [Graph500 Benchmark](https://graph500.org/)
- [OpenMPI Documentation](https://www.open-mpi.org/doc/)
