#include <bits/stdc++.h>

using namespace std;

const int BATCH_SIZE = 20;

struct xorshift {
  unsigned x, y, z, w;

  xorshift(): x(123456789), y(38012123), z(7777777), w(8392032) {}

  unsigned next() {
    unsigned t = x ^ (x<<11);

    x = y; y = z; z = w;

    return w = w ^ (w>>19) ^ t ^ (t>>8);
  }
} rng;

double rand_01() {
  return rng.next()%1000000001 / 1000000000.0;
}

struct matrix {
  int n, m;
  double **a;

  matrix(): n(0), m(0), a(nullptr) {}
  matrix(int rows, int cols) {
    int i, j;

    n = rows;
    m = cols;
    
    a = new double* [n];
    for(i=0;i<n;i++) {
      a[i] = new double [m];
      
      for(j=0;j<m;j++) {
        a[i][j] = 0.0;
      }
    }
  }
  matrix(const matrix &x) {
    int i, j;

    n = x.n;
    m = x.m;

    a = new double* [n];
    for(i=0;i<n;i++) {
      a[i] = new double [m];

      for(j=0;j<m;j++) {
        a[i][j] = x.a[i][j];
      }
    }
  }

  ~matrix() {
    for(int i=0;i<n;i++) {
      free(a[i]);
    }

    free(a);
  }

  void operator =(const matrix &x) {
    int i, j;

    n = x.n;
    m = x.m;

    a = new double* [n];
    for(i=0;i<n;i++) {
      a[i] = new double [m];

      for(j=0;j<m;j++) {
        a[i][j] = x.a[i][j];
      }
    }
  }

  void randomize() {
    int i, j;

    for(i=0;i<n;i++) {
      for(j=0;j<m;j++) {
        a[i][j] = rand_01();
        
        if(rng.next()%2==0) a[i][j] = -a[i][j];
      }
    }
  }

  void zero() {
    int i, j;

    for(i=0;i<n;i++) {
      for(j=0;j<m;j++) {
        a[i][j] = 0.0;
      }
    }
  }

  //Assume that X has the same dimensions
  void add(matrix x) {
    int i, j;

    for(i=0;i<n;i++) {
      for(j=0;j<m;j++) {
        a[i][j] += x[i][j];
      }
    }
  }

  double* &operator [](const int &idx) {
    return a[idx];
  }
};

//Assume A and B have the same dimensions
matrix add(matrix a, matrix b) {
  matrix ans(a.n, a.m);
  int i, j;

  for(i=0;i<a.n;i++) {
    for(j=0;j<a.m;j++) {
      ans[i][j] = a[i][j] + b[i][j];
    }
  }

  return ans;
}

//Assume A and B have the same dimensions
matrix subtract(matrix a, matrix b) {
  matrix ans(a.n, a.m);
  int i, j;

  for(i=0;i<a.n;i++) {
    for(j=0;j<a.m;j++) {
      ans[i][j] = a[i][j] - b[i][j];
    }
  }

  return ans;
}

//Assume A and B have the same dimensions
matrix term_by_term(matrix a, matrix b) {
  matrix ans(a.n, a.m);
  int i, j;

  for(i=0;i<a.n;i++) {
    for(j=0;j<a.m;j++) {
      ans[i][j] = a[i][j] * b[i][j];
    }
  }

  return ans;
}

matrix transpose(matrix a) {
  matrix ans(a.m, a.n);
  int i, j;

  for(i=0;i<a.n;i++) {
    for(j=0;j<a.m;j++) {
      ans[j][i] = a[i][j];
    }
  }

  return ans;
}


matrix multiply(matrix a, matrix b) {
  matrix ans(a.n, b.m);
  int i, j, z;

  for(i=0;i<a.n;i++) {
    for(j=0;j<b.m;j++) {
      for(z=0;z<a.m;z++) {
        ans[i][j] += a[i][z] * b[z][j];
      }
    }
  }

  return ans;
}

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
  return x * (1.0 - x);
}

matrix sigmoid(matrix a) {
  matrix ans(a.n, a.m);
  int i, j;

  for(i=0;i<a.n;i++) {
    for(j=0;j<a.m;j++) {
      ans[i][j] = sigmoid(a[i][j]);
    }
  }

  return ans;
}

matrix sigmoid_derivative(matrix a) {
  matrix ans(a.n, a.m);
  int i, j;

  for(i=0;i<a.n;i++) {
    for(j=0;j<a.m;j++) {
      ans[i][j] = sigmoid_derivative(a[i][j]);
    }
  }

  return ans;
}

struct neural_network {
  int n;
  vector < int > size;
  vector < matrix > w, b, delta_w, delta_b;
  double learning_rate;

  neural_network(){}
  neural_network(vector < int > sz, double alpha) {
    int i;

    n = (int)(sz.size());
    size = sz;

    w.resize(n - 1);
    b.resize(n - 1);
    delta_w.resize(n - 1);
    delta_b.resize(n - 1);

    for(i=0;i<n-1;i++) {
      w[i] = matrix(size[i], size[i + 1]);
      b[i] = matrix(1, size[i + 1]);
      delta_w[i] = matrix(size[i], size[i + 1]);
      delta_b[i] = matrix(1, size[i + 1]);

      w[i].randomize();
      b[i].randomize();
    }

    learning_rate = alpha;
  }

  matrix feedforward(matrix input) {
    int i;

    for(i=0;i<n-1;i++) {
      input = sigmoid(add(multiply(input, w[i]), b[i]));
    }

    return input;
  }

  void backpropagation(matrix input, matrix output) {
    vector < matrix > l;
    matrix delta;
    int i;

    l.push_back(input);
    for(i=0;i<n-1;i++) {
      input = sigmoid(add(multiply(input, w[i]), b[i]));
      l.push_back(input);
    }

    delta = term_by_term(subtract(input, output), sigmoid_derivative(l[n - 1]));
    
    delta_b[n - 2].add(delta);
    delta_w[n - 2].add(multiply(transpose(l[n - 2]), delta));

    for(i=n-3;i>=0;i--) {
      delta = multiply(delta, transpose(w[i + 1]));

      delta = term_by_term(delta, sigmoid_derivative(l[i + 1]));

      delta_b[i].add(delta);
      delta_w[i].add(multiply(transpose(l[i]), delta));
    }
  }

  void train(vector < matrix > inputs, vector < matrix > outputs) {
    int i, j, z;

    for(i=0;i<n-1;i++) {
      delta_w[i].zero();
      delta_b[i].zero();
    }

    for(i=0;i<(int)(inputs.size());i++) {
      backpropagation(inputs[i], outputs[i]);
    }

    for(i=0;i<n-1;i++) {
      for(j=0;j<delta_w[i].n;j++) {
        for(z=0;z<delta_w[i].m;z++) {
          delta_w[i][j][z] /= (double)(inputs.size());
          w[i][j][z] -= learning_rate * delta_w[i][j][z];
        }
      }

      for(j=0;j<delta_b[i].n;i++) {
        for(z=0;z<delta_b[i].m;z++) {
          delta_b[i][j][z] /= (double)(inputs.size());
          b[i][j][z] -= learning_rate * delta_b[i][j][z];
        }
      }
    }
  }
};

vector < matrix > train_input, train_output;
neural_network net({784, 20, 10}, 1.0);

vector < int > split(string s) {
  int i, curr = 0;
  vector < int > ans;

  for(i=0;i<(int)(s.size());i++) {
    if(s[i]==',') {
      ans.push_back(curr);
      curr = 0;
    }
    else {
      curr *= 10;
      curr += s[i] - '0';
    }
  }

  ans.push_back(curr);

  return ans;
}

void time_stamp() {
  cerr<<"Time: "<<(int)(clock() * 1000.0 / CLOCKS_PER_SEC)<<" ms."<<endl;
}

void parse_training_data() {
  ifstream IN("train.csv");
  int i, j;
  string trash;
  vector < int > v;
  matrix input(1, 784), output(1, 10);

  train_input.reserve(42000);
  train_output.reserve(42000);

  IN>>trash;
  for(i=0;i<42000;i++) {
    IN>>trash;

    v = split(trash);

    output.zero();
    output[0][v[0]] = 1.0;

    for(j=1;j<785;j++) {
      input[0][j - 1] = v[j] / 255.0;
    }

    train_input.push_back(input);
    train_output.push_back(output);
  }
  
  cerr<<"Training data loaded!"<<endl;
  time_stamp();
}

void random_shuffle(vector < int > &v) {
  for(int i=(int)(v.size())-1;i>=0;i--) {
    swap(v[i], v[rng.next() % (i + 1)]);
  }
}

void train() {
  int i, j, epoch;
  vector < int > idx;
  vector < matrix > inputs, outputs;
  double error;
  matrix curr_output;

  for(i=0;i<42000;i++) {
    idx.push_back(i);
  }

  for(epoch=1;epoch<=10;epoch++) {
    cerr<<"Epoch "<<epoch<<" starting."<<endl;

    error = 0.0;

    random_shuffle(idx);
    for(i=0;i<42000;i+=BATCH_SIZE) {
      inputs.clear();
      outputs.clear();

      for(j=0;j<BATCH_SIZE;j++) {
        inputs.push_back(train_input[idx[i + j]]);
        outputs.push_back(train_output[idx[i + j]]);
      }

      net.train(inputs, outputs);
    }

    for(i=0;i<42000;i++) {
      curr_output = net.feedforward(train_input[i]);

      for(j=0;j<10;j++) {
        error += (curr_output[0][j] - train_output[i][0][j]) * (curr_output[0][j] - train_output[i][0][j]);
      }
    }

    error /= 10.0;
    error /= 42000.0;

    cerr<<"Epoch "<<epoch<<" finished."<<endl;
    cerr<<"Error: "<<error<<endl;
    time_stamp();
    cerr<<endl;
  }
}

void test() {
  ifstream IN("test.csv");
  ofstream OUT("ans.csv");
  string trash;
  vector < int > v;
  int i, j, idx;
  double max_value;
  matrix curr_input(1, 784), curr_output;

  OUT<<"ImageId,Label"<<endl;

  IN>>trash;
  
  for(i=0;i<28000;i++) {
    IN>>trash;
    v = split(trash);

    for(j=0;j<784;j++) {
      curr_input[0][j] = v[j] / 255.0;
    }

    curr_output = net.feedforward(curr_input);
    
    max_value = -1;
    for(j=0;j<10;j++) {
      if(curr_output[0][j]>max_value) {
        max_value = curr_output[0][j];
        idx = j;
      }
    }

    OUT<<i+1<<","<<idx<<endl;
  }

  OUT.close();
}

int main() {
  parse_training_data();
  train();
  test();

  return 0;
}
