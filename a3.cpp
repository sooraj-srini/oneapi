#include <vector>
#include <sycl/sycl.hpp>

using namespace sycl;

int main()
{

    // take in input from stdin
    std::ios::sync_with_stdio(false);

    int num_meshes, R, C;
    std::cin >> num_meshes >> R >> C;
    std::vector<int> sx(num_meshes), sy(num_meshes), opacity(num_meshes);
    std::vector<std::pair<int, int>> x(num_meshes); 
    std::vector<std::vector<int>> mesh(num_meshes);

    for (int i = 0; i < num_meshes; i++)
    {
        std::cin >> sx[i] >> sy[i] >> x[i].first >> x[i].second >> opacity[i];
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
    hOffset[num_meshes] = offset;

    int num_trans;
    std::cin >> num_trans;
    std::vector<std::pair<int, int>> up(num_meshes, {0, 0});
    for (int i = 0; i < num_trans; i++)
    {
        int v, c, a;
        std::cin >> v >> c >> a;
        if (c == 0)
            up[v].first -= a;
        else if (c == 1)
            up[v].first += a;
        else if (c == 2)
            up[v].second -= a;
        else if (c == 3)
            up[v].second += a;
    }
    std::vector<std::pair<int, int>> cum_up(up);

    // input taken in hopefully

    // we need to do dfs on the tree and update the transformations of each node
    std::vector<int> parent(num_meshes, -1);
    std::vector<int> grandparent(num_meshes, -1);
    std::vector<int64_t> grid(R*C, 0);
	auto start = std::chrono::high_resolution_clock::now () ;
    queue h;

    {
        buffer hOffset_buf(hOffset), hCsr_buf(hCsr), sx_buf(sx), sy_buf(sy), opacity_buf(opacity);
        buffer parent_buf(parent), grandparent_buf(grandparent);
        buffer cum_up_buf(cum_up), up_buf(up), x_buf(x);

        h.submit([&](handler &hh)
                 {
                     accessor hOffset_acc(hOffset_buf, hh, read_only);
                     accessor hCsr_acc(hCsr_buf, hh, read_only);
                     accessor parent_acc(parent_buf, hh, write_only);

                     // update parent of node in parent accessor

                     hh.parallel_for(num_meshes, [=](id<1> i)
                                     {
                            int start = hOffset_acc[i];
                            int end = hOffset_acc[i + 1];
                            for (int j = start; j < end; j++)
                            {
                                parent_acc[hCsr_acc[j]] = i;
                            } }); });
        int updates = 32 - __builtin_clz(V);

        for (int i = 0; i < updates; i++)
        {

            h.submit([&](handler &hh){
                accessor parent_acc(parent_buf, hh, read_write);
                accessor grand_acc(grandparent_buf, hh, read_write);
                if(i% 2 == 1) {
                    std::swap(parent_acc, grand_acc);
                }
                accessor up_acc(up_buf, hh, read_only);
                accessor cum_up_acc(cum_up_buf, hh, read_write);

                 // update parent of node in parent accessor

                hh.parallel_for(num_meshes, [=](id<1> i)
                    {
                    int par = parent_acc[i];
                    if(par >= 0) {
                        cum_up_acc[i].first = cum_up_acc[i].first + up_acc[par].first;
                        cum_up_acc[i].second = cum_up_acc[i].second + up_acc[par].second;
                        grand_acc[i] = parent_acc[par];
                    } else {
                        grand_acc[i] = -1;
                    } 
                    }
                ); 
            });

            h.submit([&](handler &hh) {
                accessor up_acc (up_buf, hh, write_only);
                accessor cum_up_acc (cum_up_buf, hh, read_only);
                hh.parallel_for(num_meshes, [=](id<1> i) {
                    up_acc[i] = cum_up_acc[i];
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
        buffer mesh_buf(meshall), offset_buf(offset); 
        buffer grid_buf(grid); 
        h.submit([&](handler &hh) {
            accessor mesh_acc(mesh_buf, hh, read_only);
            accessor offset_acc(offset_buf, hh, read_only);
            accessor cum_up_acc(cum_up_buf, hh, read_only);
            accessor sx_acc(sx_buf, hh, read_only);
            accessor sy_acc(sy_buf, hh, read_only);
            accessor x_acc(x_buf, hh, read_only);
            accessor op_acc(opacity_buf, hh, read_only);
            accessor grid_acc(grid_buf, hh, read_write);

            hh.parallel_for(total, [=](id<1> i) {
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
                int ux = x_acc[vertex].first + r + cum_up_acc[vertex].first;
                int uy = x_acc[vertex].second + c + cum_up_acc[vertex].second;

                if (ux >= 0 && ux < R && uy >= 0 && uy < C) {
                    int idx = ux * C + uy;
                    atomic_ref<int64_t, memory_order::relaxed, memory_scope::system, access::address_space::global_space> lock_ref(grid_acc[idx]);
                    int64_t op = op_acc[vertex];
                    int64_t val = mesh_acc[offset_acc[vertex] + r*sy_acc[vertex] + c];
                    int64_t both = (op << 32) | val;
                    lock_ref.fetch_max(both);
                }
                    
            });

        });

        h.submit([&](handler &hh) {
            accessor grid_acc(grid_buf, hh, read_write);
            hh.parallel_for(R*C, [=](id<1> i) {
                int idx = i[0];
                int64_t both = grid_acc[idx];
                int val = both & 0xFFFFFFFF;
                grid_acc[idx] = val;
            });
        });

    }
	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::milli> timeTaken = end-start;

	// printf ("execution time : %f\n", timeTaken) ;
    std::cerr << "Execution time: " << timeTaken.count() << " milliseconds" << std::endl;

    // now we have the parent of each node
    // print the parents

    for(int i=0; i<R; i++) {
        for(int j=0; j<C; j++) {
            std::cout << grid[i*C + j] << " ";

        }
        std::cout << "\n";
    }
}