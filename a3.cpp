#include <algorithm>
#include <vector>
#include <CL/sycl.hpp>

int main()
{

    // take in input from stdin

    int num_meshes, R, C;
    std::cin >> num_meshes >> R >> C;
    std::vector<int> x(num_meshes), y(num_meshes), sx(num_meshes), sy(num_meshes), opacity(num_meshes);
    std::vector<std::vector<int>> mesh(num_meshes);

    for (int i = 0; i < num_meshes; i++)
    {
        std::cin >> sx[i] >> sy[i] >> x[i] >> y[i] >> opacity[i];
        mesh[i].resize(sx[i] * sy[i]);
        for (int j = 0; j < sx[i] * sy[i]; j++)
        {
            std::cin >> mesh[i][j];
        }
    }

    int E, V = num_meshes;
    std::cin >> E;
    std::vector<std::vector<int>> graph(num_meshes + 1);
    for (int i = 0; i < E; i++)
    {
        int u, v;
        std::cin >> u >> v;
        graph[u].push_back(v);
    }
    std::vector<int> hOffset(num_meshes + 1), hCsr(E);
    int offset = 0;
    for (int i = 0; i < num_meshes; i++)
    {
        hOffset[i] = offset;
        for (int j = 0; j < graph[i].size(); j++)
        {
            hCsr[offset++] = graph[i][j];
        }
    }

    int num_trans;
    std::cin >> num_trans;
    std::vector<int> up(num_meshes), right(num_meshes);
    for (int i = 0; i < num_trans; i++)
    {
        int v, c, a;
        std::cin >> v >> c >> a;
        if (c == 0)
            up[v] -= a;
        else if (c == 1)
            up[v] += a;
        else if (c == 2)
            right[v] -= a;
        else if (c == 3)
            right[v] += a;
    }
    std::vector<int> cum_up(up), cum_right(right);

    // input taken in hopefully

    // we need to do dfs on the tree and update the transformations of each node
    std::vector<int> parent(num_meshes, -1);
    std::vector<int> grandparent(num_meshes, -1);
    std::vector<int> grid(R*C, 0);
    sycl::queue h;

    {
        sycl::buffer hOffset_buf(hOffset), hCsr_buf(hCsr), up_buf(up), right_buf(right), x_buf(x), y_buf(y), sx_buf(sx), sy_buf(sy), opacity_buf(opacity);
        sycl::buffer parent_buf(parent), grandparent_buf(grandparent);
        sycl::buffer cum_up_buf(cum_up), cum_right_buf(cum_right);

        h.submit([&](sycl::handler &hh)
                 {
                     sycl::accessor hOffset_acc(hOffset_buf, hh, sycl::read_only);
                     sycl::accessor hCsr_acc(hCsr_buf, hh, sycl::read_only);
                     sycl::accessor parent_acc(parent_buf, hh, sycl::write_only);

                     // update parent of node in parent accessor

                     hh.parallel_for(num_meshes, [=](sycl::id<1> i)
                                     {
                            int start = hOffset_acc[i];
                            int end = hOffset_acc[i + 1];
                            for (int j = start; j < end; j++)
                            {
                                if (parent_acc[hCsr_acc[j]] == -1)
                                {
                                    parent_acc[hCsr_acc[j]] = i;
                                }
                            } }); });
        int updates = 32 - __builtin_clz(V);

        for (int i = 0; i < updates; i++)
        {

            h.submit([&](sycl::handler &hh)
                     {
                        sycl::accessor parent_acc(parent_buf, hh, sycl::read_write);
                        sycl::accessor grand_acc(grandparent_buf, hh, sycl::read_write);
                        if(i% 2 == 1) {
                            auto tmp = parent_acc;
                            parent_acc = grand_acc;
                            grand_acc = tmp;
                        }
                        sycl::accessor up_acc(up_buf, hh, sycl::read_only);
                        sycl::accessor right_acc(right_buf, hh, sycl::read_only);
                        sycl::accessor cum_up_acc(cum_up_buf, hh, sycl::read_write);
                        sycl::accessor cum_right_acc(cum_right_buf, hh, sycl::read_write);

                         // update parent of node in parent accessor

                        hh.parallel_for(num_meshes, [=](sycl::id<1> i)
                            {
                            int par = parent_acc[i];
                            if(par >= 0) {
                                cum_up_acc[i] = cum_up_acc[i] + up_acc[par];
                                cum_right_acc[i] = cum_right_acc[i] + right_acc[par];
                                grand_acc[i] = parent_acc[par];
                            } else {
                                grand_acc[i] = -1;
                            } 
                            }
                        ); 
                        }
                        );

            h.submit([&](sycl::handler &hh) {
                sycl::accessor up_acc (up_buf, hh, sycl::write_only);
                sycl::accessor right_acc (right_buf, hh, sycl::write_only);
                sycl::accessor cum_right_acc (cum_right_buf, hh, sycl::read_only);
                sycl::accessor cum_up_acc (cum_up_buf, hh, sycl::read_only);
                hh.parallel_for(num_meshes, [=](sycl::id<1> i) {
                    up_acc[i] = cum_up_acc[i];
                    right_acc[i] = cum_right_acc[i];
                });
            });

        }

        //update mesh
        std::vector<int> offset(V);
        offset[0] = 0;
        for(int i=1; i<V; i++) {
            offset[i] += offset[i-1] + sx[i-1]*sy[i-1];
        }
        int total = offset[V-1] + sx[V-1]*sy[V-1];
        std::vector<int> meshall(total);
        for(int i=0; i<V; i++){
            std::copy(mesh[i].begin(), mesh[i].end(), meshall.begin() + offset[i]);
        }
        sycl::buffer mesh_buf(meshall), offset_buf(offset);
        std::vector<int> lock(R*C, 0);
        sycl::buffer lock_buf(lock); 
        sycl::buffer grid_buf(grid); 
        std::cout << total << std::endl;
        h.submit([&](sycl::handler &hh) {
            sycl::accessor mesh_acc(mesh_buf, hh, sycl::read_only);
            sycl::accessor offset_acc(offset_buf, hh, sycl::read_only);
            sycl::accessor cum_up_acc(cum_up_buf, hh, sycl::read_only);
            sycl::accessor cum_right_acc(cum_right_buf, hh, sycl::read_only);
            sycl::accessor sx_acc(sx_buf, hh, sycl::read_only);
            sycl::accessor sy_acc(sy_buf, hh, sycl::read_only);
            sycl::accessor x_acc(x_buf, hh, sycl::read_only);
            sycl::accessor y_acc(y_buf, hh, sycl::read_only);
            sycl::accessor op_acc(opacity_buf, hh, sycl::read_only);
            sycl::accessor grid_acc(grid_buf, hh, sycl::read_write);
            sycl::accessor lock_acc(lock_buf, hh, sycl::read_write);

            hh.parallel_for(total, [=](sycl::id<1> i) {
                int tid = i[0];
                int left = 0, right = V;
                while (left < right) {
                    int mid = (left + right)/2;
                    if(offset_acc[mid] <= tid) {
                        if(mid == V-1 || offset_acc[mid + 1] > tid) {
                            break;
                        }
                        else if(offset_acc[mid + 1] <= tid) {
                            left = mid + 1;
                        }
                    } else {
                        right = mid;
                    }
                }
                int vertex = (left + right)/2;
                int off = tid - offset_acc[vertex];
                int r =  off /sy_acc[vertex];
                int c = off % sy_acc[vertex];
                if (vertex < V && r < sx_acc[vertex] && c < sy_acc[vertex]) {
                    int ux = x_acc[vertex] + r + cum_up_acc[vertex];
                    int uy = y_acc[vertex] + c + cum_right_acc[vertex];

                    if (ux >= 0 && ux < R && uy >= 0 && uy < C) {
                        int idx = ux * C + uy;
                        int op = op_acc[vertex];
                        bool done = false, updated = false;
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space> lock_ref(lock_acc[idx]);
                        do {
                            done = updated;
                            int old = lock_ref;
                            bool val = !(old >= 0 && old < op);
                            int exp = old - INT_MAX*val;
                            int des = -1;
                            bool suc = lock_ref.compare_exchange_strong(exp, des);

                            if(suc) {
                                grid_acc[idx] = mesh_acc[offset_acc[vertex] + r*sy_acc[vertex] + c];
                                lock_ref = op;
                                updated = true;
                            }
                            updated = updated | (old >= op);

                        } while(!done);
                    }
                    
                }
            });

        });



    }

    // now we have the parent of each node
    // print the parents

    for(int i=0; i<R; i++) {
        for(int j=0; j<C; j++) {
            std::cout << grid[i*C + j] << " ";

        }
        std::cout << "\n";
    }
}