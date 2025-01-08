# Julia server that listens for commands and evaluates them

# This is a very simple server program that listens at some port for commands
# provided as simple strings, executes them, and returns the result as a string.

# In the context of tuning with irace, it provides the somtimes big advantage
# that individual function calls for evaluation do not need to start the whole
# Julia runtime, which can be quite slow. Instead, fast function calls can be
# provided to this server via some fast socket communication.


include("../julia-function-to-tune.jl")  # adapt to make your application available


using Sockets

const port = 9375

function handle_client(client::TCPSocket)
    try
        while isopen(client)
            # Read data from the client (up to a reasonable limit)
            data = readuntil(client, '\n', keep=true)
            if isempty(data)
                break  # Client disconnected
            end

            request_str = String(data)
            println("Received: ", request_str)

            try
                expr = Meta.parse(request_str)
                result = eval(expr)
                println("result $result")
                println(client, result)
                flush(client)
            catch e
                result = "Error evaluating expression: $(e)"
            end
            break  # We just execute one function call here
        end
    catch e
        println("Client error: ", e)
    finally
        close(client)
        println("Client disconnected.")
    end
end


function start_server(port::Integer=port)
    server = listen(port)
    println("Server listening on port ", port)

    try
        while true
            client = accept(server)
            println("Client connected.")
            # @async handle_client(client) # Handle each client in a separate task
            handle_client(client) # Handle each client in a separate task
        end
    catch e
        println("Server error: ", e)
    finally
        close(server)
    end
end

# This would be a function for using the server from Julia, which, however, is not necessary
#
# function send_command(command::String; host::String="localhost", port::Integer=9375)
#     try
#         socket = connect(host, port)
#         write(socket, command)
#         flush(socket)
#         response = readuntil(socket, '\n')
#         println("got response: ", response)
#         close(socket)
#         return response
#     catch e
#         println("Error: ", e)
#         return Dict("status" => "error", "message" => string(e))
#     end
# end

start_server()
# To stop the server gracefully, you'll need to interrupt the Julia process (e.g., Ctrl+C).
