using Serialization

function save(data, file)
    open(file, "w") do f
        serialize(f, data)
    end
end

function load(file)
    data = open(file, "r") do f
        deserialize(f)
    end
    return data
end


using PythonCall

pickle = pyimport("pickle")

function loadpickle(datafile)
    data = pywith(pybuiltins.open(datafile, "rb")) do f
        pickle.load(f)
    end

    return data
end

function savepickle(data, datafile)
    pywith(pybuiltins.open(datafile, "wb")) do f
        pickle.dump(data, f)
    end    
end

