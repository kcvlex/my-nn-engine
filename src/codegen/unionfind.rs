use std::collections::HashMap;
use std::hash::Hash;

pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![1; n],
        }
    }

    fn representative(&mut self, x: usize) -> usize {
        if self.parent[x] == x {
            x
        } else {
            let p = self.representative(self.parent[x]);
            self.parent[x] = p;
            p
        }
    }

    fn merge(&mut self, x: usize, y: usize) {
        let x = self.representative(x);
        let y = self.representative(y);
        if x == y {
            return;
        }

        let (x, y) = if self.rank[x] < self.rank[y] {
            (x, y)
        } else {
            (y, x)
        };
        self.parent[x] = y;
        if self.rank[x] == self.rank[y] {
            self.rank[y] += 1;
        }
    }
}

pub struct HashUnionFind<T: Eq + Hash + Clone> {
    uf: UnionFind,
    map: HashMap<T, usize>,
    inv: Vec<T>,
}

impl<T: Eq + Hash + Clone> HashUnionFind<T> {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            uf: UnionFind::with_capacity(n),
            map: HashMap::with_capacity(n),
            inv: Vec::with_capacity(n),
        }
    }

    pub fn append(&mut self, value: T) {
        if self.map.len() == self.uf.parent.len() {
            panic!()
        }
        let id = self.map.len();
        self.map.insert(value.clone(), id);
        self.inv.push(value);
    }

    pub fn representative(&mut self, value: &T) -> T {
        let id = self.map.get(value).unwrap();
        let id = self.uf.representative(*id);
        self.inv[id].clone()
    }

    pub fn merge(&mut self, x: &T, y: &T) {
        let x = self.map.get(x).unwrap();
        let y = self.map.get(y).unwrap();
        self.uf.merge(*x, *y);
    }
}
