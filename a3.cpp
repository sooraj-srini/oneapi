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
    for (int i = 0; i < num_meshes; i++)
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
    sycl::queue h;

    {
        sycl::buffer hOffset_buf(hOffset), hCsr_buf(hCsr), up_buf(up), right_buf(right), x_buf(x), y_buf(y), sx_buf(sx), sy_buf(sy), opacity_buf(opacity);
        sycl::buffer parent_buf(parent), grandparent_buf(grandparent);

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
                            } 
                        });

                }
            );
    }

    // now we have the parent of each node
    // print the parents

    for (int i = 0; i < num_meshes; i++)
    {
        std::cout << parent[i] << " ";
    }
}