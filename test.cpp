#include <bits/stdc++.h>
// #include <time.h>
#include <atcoder/all>

using namespace std;
using namespace atcoder;

typedef long long ll;
typedef long double ld;
typedef vector<string> vs;
typedef vector<vector<string>> vvs;
typedef vector<double> vd;
typedef vector<vector<double>> vvd;
typedef tuple<int, int, int> tiii;
typedef vector<pair<ll, ll>> vpl;
typedef pair<string, string> ps;
typedef pair<bool, bool> pb;
typedef queue<pair<int, int>> que_pi;
typedef queue<pair<ll, ll>> que_pl;

template <class T> using max_heap = priority_queue<T>;
template <class T> using min_heap = priority_queue<T, vector<T>, greater<>>;

#define all(x) (x).begin(), (x).end()

#define ForEach(it, c) for (__typeof(c).begin() it = (c).begin(); it != (c).end(); it++)

#define per(i, n) for (int i = n; i >= 0; --i)
#define per2(i, n, x) for (int i = n; i >= x; --i)
#define srep(i, s, t) for (int i = s; i < (t); ++i)

#define MOD1000000007 1000000007

int chtoi(char ch) {
  return ch - '0';
}

template <typename T> bool chmax(T &a, const T &b) {
  if (a < b) {
    a = b;
    return true;
  }
  return false;
}

template <typename T> bool chmin(T &a, const T &b) {
  if (a > b) {
    a = b; // aをbで更新
    return true;
  }
  return false;
}

template <typename T> ostream &operator<<(ostream &os, const vector<T> &v) {
  for (int i = 0; i < (int)v.size(); i++) {
    os << v[i] << (i + 1 != v.size() ? " " : "");
  }
  return os;
}

template <typename T> istream &operator>>(istream &is, vector<T> &v) {
  for (T &in : v)
    is >> in;
  return is;
}

int rot(int x) {
  int tot = 1;
  while (x >= tot * 10)
    tot *= 10;
  return x / 10 + x % 10 * tot;
}

int rota(int x) {
  int tot = 1;
  while (x > tot * 10)
    tot *= 10;
  return x % 10 * tot + x / 10;
}

string toBinary(ll n) {
  string r;
  while (n != 0) {
    r += (n % 2 == 0 ? "0" : "1");
    n /= 2;
  }
  reverse(r.begin(), r.end());
  return r;
}

ll LL_INF = 1LL << 55;
int INT_INF = 2LL << 9;

class UnionFind {
private:
  vector<int> rank, p;
  void link(int x, int y) {
    if (rank[x] > rank[y]) swap(x, y);
    p[x] = y;
    if (rank[x] == rank[y]) rank[y]++;
  }

public:
  UnionFind(int n) : rank(n), p(n) {
    for (int i = 0; i < n; i++)
      p[i] = i, rank[i] = 0;
  }
  void Union(int x, int y) {
    if (Find(x) != Find(y)) link(Find(x), Find(y));
  }
  int Find(int x) {
    return (x != p[x] ? p[x] = Find(p[x]) : p[x]);
  }
  bool Same(int x, int y) {
    return (Find(x) == Find(y));
  }
};

template <class T> class SGraph {
protected:
  struct edge {
    int u, v;
    T cost;
    bool operator<(const edge &other) const {
      return cost < other.cost;
    }
  };
  typedef vector<edge> Edges;
  Edges edges;

public:
  SGraph<T>(){};
  void add_edge(int u, int v, T cost) {
    edges.push_back((edge){u, v, cost});
  }
};

template <class T> class MinimumSpanningTree : public UnionFind, public SGraph<T> {
public:
  MinimumSpanningTree(int n) : SGraph<T>(), UnionFind(n){};
  T run() {
    sort(SGraph<T>::edges.begin(), SGraph<T>::edges.end());
    T ret = 0;
    ForEach(it, SGraph<T>::edges) {
      if (!Same(it->u, it->v)) {
        Union(it->u, it->v);
        ret += it->cost;
      }
    }
    return ret;
  }
};

ll modinv(ll a, ll m) {
  ll b = m, u = 1, v = 0;
  while (b) {
    ll t = a / b;
    a -= t * b;
    swap(a, b);
    u -= t * v;
    swap(u, v);
  }
  u %= m;
  if (u < 0) u += m;
  return u;
}

ll mod = 998244353;

bool check(int tmp) {
  if (1 <= tmp && tmp <= 6)
    return true;
  else
    return false;
}

bool check(ll a, ll b, ll n) {
  return (double)a + b >= (double)n / (a * a + b * b);
}

ll f(ll a, ll b) {
  return a * a * a + a * a * b + a * b * b + b * b * b;
}

int ret_max(string s) {
  int ret = 0;
  for (char ch : s) {
    chmax(ret, ch - '0');
  }
  return ret;
}

int s(string s) {
  int sum = 0;
  for (auto ch : s) {
    sum += ch - '0';
  }
  return sum;
}

double get_descent(int i, int j, vector<pair<int, int>> &vec) {
  double x1 = vec[i].first;
  double y1 = vec[i].second;
  double x2 = vec[j].first;
  double y2 = vec[j].second;

  return (y2 - y1) / (x2 - x1);
}

struct info {
  ll aoki, takahashi, sum;
};

bool comp(info &tmp1, info &tmp2) {
  if (tmp1.sum > tmp2.sum) {
    return true;
  } else if (tmp1.sum == tmp2.sum) {
    if (tmp1.aoki >= tmp2.aoki) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

int inv(int a, int M) {
  return a == 1 ? 1 : (M + (1 - (long long)M * inv(M % a, a)) / a);
}
int gcd(int p, int q) {
  while (q) {
    int t = p % q;
    p = q;
    q = t;
  }
  return p;
}

ll modPow(ll a, ll n, ll mod) {
  if (mod == 1) return 0;
  ll ret = 1;
  ll p = a % mod;
  while (n) {
    if (n & 1) ret = ret * p % mod;
    p = p * p % mod;
    n >>= 1;
  }
  return ret;
}

const int dx[4] = {1, 0, -1, 0};
const int dy[4] = {0, 1, 0, -1};

bool is_in_matrix(int x, int y, int w, int h) {
  return (0 <= x && x < w && 0 <= y && y < h);
}

template <class T> int LIS(vector<T> a, bool is_strong = true) {
  const T INF = 1 << 30; // to be set appropriately
  int n = (int)a.size();
  vector<T> dp(n, INF);
  for (int i = 0; i < n; ++i) {
    if (is_strong)
      *lower_bound(dp.begin(), dp.end(), a[i]) = a[i];
    else
      *upper_bound(dp.begin(), dp.end(), a[i]) = a[i];
  }
  return lower_bound(dp.begin(), dp.end(), INF) - dp.begin();
}

struct edge {
  /* data */
  int to, cost, idx;
};

void dijkstra(int s, vector<int> &d, vector<vector<edge>> &G) {
  d[s] = 0;
  priority_queue<pair<int, int>> Q;
  Q.push(make_pair(0, s));
  while (!Q.empty()) {
    pair<int, int> p = Q.top();
    Q.pop();
    int pos = p.second, cost = -p.first;
    if (cost > d[pos]) continue;
    for (int i = 0; i < G[pos].size(); i++) {
      edge e = G[pos][i];
      int to = e.to;
      int newcost = cost + e.cost;
      if (newcost < d[to]) {
        d[to] = newcost;
        Q.push(make_pair(-d[to], to));
      }
    }
  }
}

bool bellman_ford(int s, vector<int> &d, vector<vector<edge>> &G) { // nは頂点数、sは開始頂点
  d[s] = 0;                                                         // 開始点の距離は0
  int n = (int)d.size();
  for (int i = 0; i < n; i++) {
    for (int v = 0; v < n; v++) {
      for (int k = 0; k < G[v].size(); k++) {
        edge e = G[v][k];
        if (d[v] != INT_INF && d[e.to] > d[v] + e.cost) {
          d[e.to] = d[v] + e.cost;
          if (i == n - 1) return false; // n回目にも更新があるなら負の閉路が存在
        }
      }
    }
  }
  return true;
}

bool warshall_floyd(int V, vector<vector<ll>> &dp) {
  for (int i = 0; i < V; i++) {
    dp[i][i] = 0;
  }
  for (int k = 0; k < V; k++) {
    for (int i = 0; i < V; i++) {
      for (int j = 0; j < V; j++) {
        chmin(dp[i][j], dp[i][k] + dp[k][j]);
      }
    }
  }
  for (int i = 0; i < V; i++) {
    if (dp[i][i] < 0) return true;
  }
  return false;
}

int matrixchainmultiplication(int n, vector<int> &p, vector<vector<int>> &m) {
  for (int i = 1; i <= n; i++)
    m[i][i] = 0;
  for (int l = 2; l <= n; l++) {
    for (int i = 1; i <= n - l + 1; i++) {
      int j = i + l - 1;
      m[i][j] = INT_INF;
      for (int k = i; k <= j - 1; k++) {
        m[i][j] = min(m[i][j], m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]);
      }
    }
  }
  return m[1][n];
}

class DisjointSet {
private:
  vector<int> rank, p;
  void link(int x, int y) {
    if (rank[x] > rank[y]) {
      p[y] = x;
    } else {
      p[x] = y;
      if (rank[x] == rank[y]) rank[y]++;
    }
  }

public:
  DisjointSet(int size) {
    rank.resize(size, 0);
    p.resize(size, 0);
  }
  void build() {
    for (int i = 0; i < rank.size(); i++) {
      makeSet(i);
    }
  }
  void makeSet(int x) {
    p[x] = x, rank[x] = 0;
  }
  void Union(int x, int y) {
    link(findSet(x), findSet(y));
  }
  int findSet(int x) {
    return (x != p[x] ? p[x] = findSet(p[x]) : p[x]);
  }
};

template <typename T> void vector_output(T vec) {
  for (int i = 0; i < vec.size(); i++) {
    cout << vec[i] << " \n"[i == vec.size() - 1];
  }
}

vector<bool> get_prime(int n) {
  vector<bool> prime(n + 1, true);
  if (n >= 0) prime[0] = false;
  if (n >= 1) prime[1] = false;
  for (int i = 2; i <= n; i++) {
    if (prime[i]) {
      for (int j = i * 2; j <= n; j += i)
        prime[j] = false;
    }
  }
  return prime;
}

ll get_distance(int Q, ll MOD, ll *cost) {
  ll ret = 0, b[2] = {1, 1};
  for (int i = 0; i < Q; i++) {
    cin >> b[i & 1];
    int U = b[0], V = b[1];
    if (U > V) swap(U, V);
    (ret += cost[V - 1] - cost[U - 1] + MOD) %= MOD;
  }
  ret = (ret + cost[b[(Q - 1) & 1] - 1]) % MOD;
  return ret;
}
vector<ll> get_fact_table(int table_size, ll MOD) {
  vector<ll> fact(table_size);
  fact[0] = fact[1] = 1;
  for (int i = 2; i < table_size; i++) {
    fact[i] = (i * fact[i - 1]) % MOD;
  }
  return fact;
}

vector<ll> get_inv_table(int table_size, ll MOD) {
  vector<ll> inv(table_size);
  inv[1] = 1;
  for (int i = 2; i < table_size; i++) {
    inv[i] = inv[MOD % i] * (MOD - MOD / i) % MOD;
  }
  return inv;
}

vector<ll> get_factinv_table(int table_size, ll MOD) {
  vector<ll> factinv(table_size);
  vector<ll> inv = get_inv_table(table_size, MOD);
  factinv[0] = 1;
  for (int i = 1; i < table_size; i++) {
    factinv[i] = factinv[i - 1] * inv[i] % MOD;
  }
  return factinv;
}

ll ncr(int n, int r, int table_size, ll MOD) {
  vector<ll> fact = get_fact_table(table_size, MOD);
  vector<ll> inv = get_inv_table(table_size, MOD);
  vector<ll> factinv = get_factinv_table(table_size, MOD);

  return ((n - r >= 0 && r >= 0) ? fact[n] * factinv[r] % MOD * factinv[n - r] % MOD : 0);
}

template <typename T> T get_cumulative_sum(T a) {
  int n = a.size();
  T s(n + 1, 0);
  for (int i = 0; i < n; i++) {
    s[i + 1] = s[i] + a[i];
  }
  return s;
}

template <typename T> vector<vector<T>> get_2d_cumulative_sum(vector<vector<T>> a) {
  int n = a.size();
  int m = a[0].size();
  vector<vector<T>> s(n + 1, vector<T>(m + 1, 0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      s[i + 1][j + 1] = a[i][j] + s[i + 1][j] + s[i][j + 1] - s[i][j];
    }
  }
  return s;
}

int hhmmss_to_second(string s) {

  int hour = (s[0] - '0') * 10 + (s[1] - '0');
  int minute = (s[3] - '0') * 10 + (s[4] - '0');
  int sec = (s[6] - '0') * 10 + (s[7] - '0');
  int ret = hour * 3600 + minute * 60 + sec;
  return ret;
}

ll sigma_tousa(ll a, ll d, ll n) {
  return n * (2 * a + (n - 1) * d) / 2;
}

void yes_no(bool question) {
  cout << (question ? "Yes" : "No") << endl;
}

vector<bool> get_divisor_table(int n) {
  vector<bool> table(n + 1, false);
  for (int i = 1; i <= sqrt(n); i++) {
    if (n % i == 0) {
      table[i] = true;
      table[n / i] = true;
    }
  }
  return table;
}

vector<pair<ll, ll>> factor(ll x) {
  vector<pair<ll, ll>> ans;
  for (ll i = 2; i * i <= x; i++)
    if (x % i == 0) {
      ans.push_back({i, 1});
      while ((x /= i) % i == 0)
        ans.back().second++;
    }
  if (x != 1) ans.push_back({x, 1});
  return ans;
}

template <typename T> T get_median(vector<T> &vec) {
  sort(vec.begin(), vec.end());
  int n = vec.size();
  int m = vec.size() / 2;
  if (n & 1) {
    return vec[m];
  } else {
    return (vec[m] + vec[m - 1]) / 2;
  }
}

template <typename T> bool is_median(vector<T> &vec, T t) {
  return t == get_median(vec);
}

template <typename T> T get_manhattan_distance(pair<T, T> s, pair<T, T> t) {
  return abs(s.first - t.first) + abs(s.second - t.second);
}

template <typename T> int get_num_lower(vector<T> &vec, T x) {
  int res = lower_bound(all(vec), x) - vec.begin();
  return res;
}

template <typename T> int get_num_lower_and_equal(vector<T> &vec, T x) {
  int res = upper_bound(all(vec), x) - vec.begin();
  return res;
}

template <typename T> int get_num_greater(vector<T> &vec, T x) {
  int res = vec.end() - upper_bound(all(vec), x);
  return res;
}

template <typename T> int get_num_greater_and_equal(vector<T> &vec, T x) {
  int res = vec.end() - lower_bound(all(vec), x);
  return res;
}

template <typename T> T get_nearest(vector<T> &a, T b) {
  int n = a.size();
  int pos = lower_bound(all(a), b) - a.begin();
  int diff1 = INT_MAX, diff2 = INT_MAX;
  int pos1 = min(pos, n - 1);
  int pos2 = max(0, pos - 1);
  diff1 = abs(b - a[pos1]);
  diff2 = abs(b - a[pos2]);
  // cout << min(diff1, diff2) << endl;
  if (diff1 < diff2)
    return a[pos1];
  else
    return a[pos2];
}

queue<pair<char, int>> makeRunLength(string s) {
  int n = s.size();
  queue<pair<char, int>> res;
  for (int i = 0; i < n; i++) {
    int cnt = 1;
    char c = s[i];
    while (i < n && s[i] == s[i + 1]) {
      i++, cnt++;
    }
    res.push({c, cnt});
  }
  return res;
}

template <typename T> pair<T, T> originRotate(T x, T y, T d) {
  T r = hypot(x, y);
  T theta = atan2(y, x);
  theta += d * acos(-1.0) / 180.0;

  T rotateX = cos(theta) * r;
  T rotateY = sin(theta) * r;

  return {rotateX, rotateY};
}

class Point {
private:
  long double x, y;

public:
  Point(long double x, long double y) : x(x), y(y) {}
  long double get_dist_square(Point other) {
    long double X, Y;
    X = this->x - other.x;
    Y = this->y - other.y;
    return X * X + Y * Y;
  }
  long double get_dist(Point other) {
    return sqrt(get_dist(other));
  }
};

class Circle {
private:
  Point p;
  long double r;

public:
  Circle(Point p, long double r) : p(p), r(r) {}
  bool isHasShare(Circle other) {
    long double R1 = abs(this->r - other.r);
    R1 *= R1;
    long double R2 = this->r + other.r;
    R2 *= R2;
    long double dist_square = this->p.get_dist_square(other.p);

    return (R1 <= dist_square && dist_square <= R2);
  }
  bool isOnCircle(Point point) {
    long double dist_square = this->p.get_dist_square(point);
    return (dist_square == this->r * this->r);
  }
};

void ok_ng(bool ok) {
  cout << (ok ? "OK" : "NG") << endl;
}

//---------------------------------------------------------------------------------------------------
int n, m, q, a[50], b[50], c[50], d[50];

int ans = 0;
int A[10];

void dfs(int cu = 0, int lst = 1) {
  if (cu == n) {
    int tot = 0;
    for (int i = 0; i < q; i++) {
      if (A[b[i]] - A[a[i]] == c[i]) tot += d[i];
      chmax(ans, tot);
      return;
    }
    for (int nxt = lst; nxt < m + 1; nxt++) {
      A[cu] = nxt;
      dfs(cu + 1, nxt);
    }
  }
}

int main() {
  for(int i=0;i<100;i++){
    cout << 2019*(i+1) << endl;
  }
}