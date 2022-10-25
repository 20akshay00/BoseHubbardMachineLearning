using ProgressMeter


function test()
    x = 1
    n = 10
    p = Progress(n; showspeed=true)

    for iter = 1:10
        x *= 2
        ProgressMeter.next!(p; showvalues = [(:iter, iter), (:x, x)])
        sleep(0.5)
    end
end