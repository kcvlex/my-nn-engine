use std::collections::HashMap;
use std::hash::Hash;

pub struct UnionFind<T: Eq + Hash + Clone> {
    parent: Vec<usize>,
    rank: Vec<usize>,
    map: HashMap<T, usize>,
    inv: Vec<T>,
}

impl<T: Eq + Hash + Clone> UnionFind<T> {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![1; n],
            map: HashMap::with_capacity(n),
            inv: Vec::with_capacity(n),
        }
    }

    pub fn append(&mut self, value: T) {
        if self.map.len() == self.parent.len() {
            panic!()
        }
        let id = self.map.len();
        self.map.insert(value.clone(), id);
        self.inv.push(value);
    }

    pub fn representative(&mut self, value: &T) -> T {
        let id = self.map.get(value).unwrap();
        let id = self.representative_internal(*id);
        self.inv[id].clone()
    }

    pub fn merge(&mut self, x: &T, y: &T) {
        let x = self.map.get(x).unwrap();
        let y = self.map.get(y).unwrap();
        self.merge_internal(*x, *y);
    }

    fn representative_internal(&mut self, x: usize) -> usize {
        if self.parent[x] == x {
            x
        } else {
            let p = self.representative_internal(self.parent[x]);
            self.parent[x] = p;
            p
        }
    }

    fn merge_internal(&mut self, x: usize, y: usize) {
        let x = self.representative_internal(x);
        let y = self.representative_internal(y);
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
