#include <algorithm>
#include <CL/sycl.hpp>
using namespace sycl;

int main()
{
    queue Q;
    std::cout << "Running on " << Q.get_device().get_info<info::device::name>() << "\n";

    constexpr int N = 1024;
    std::array<int, N> a, b, c;

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
        // std::cout <<a[i] <<" " << b[i] << " " << c[i] <<std::endl;
    }

    //these braces are to trigger write back from device memory to host memory
    //once the buffer goes out of scope, ig
    {
        buffer A(a), B(b), C(c);
        std::cout << "Issuing work to gpu" << std::endl;

        Q.submit([&](handler &h)
                 {
        accessor aA(A, h, read_only);
        accessor aB(B, h, read_only);
        accessor aC(C, h, write_only);

        h.parallel_for(N, [=](id<1> i) {
            aC[i] = aA[i] + aB[i];
        }); })
            .wait();
    }

    std::cout << "finsiehd work to gpu" << std::endl;

    for (int i = 0; i < N; i++)
    {
        assert(c[i] == i + i);
        // std::cout << c[i] << " ";
    }

    std::cout << "Success\n";
    return 0;
}