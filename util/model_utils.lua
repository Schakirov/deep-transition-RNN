
-- adapted from https://github.com/wojciechz/learning_to_execute
-- utilities for combining/flattening parameters in a model
-- the code in this script is more general than it needs to be, which is 
-- why it is kind of a large

require 'torch'
local model_utils = {}
function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local tn = torch.typename(layer)
	local net_params, net_grads = networks[i]:parameters()

	if net_params then
	    for _, p in pairs(net_params) do
		parameters[#parameters + 1] = p
  --  if (#parameters == 12) then parameters[#parameters] = parameters[#parameters] * 1.001 end --------------------------------------------------------------------
	    end
	    for _, g in pairs(net_grads) do
		gradParameters[#gradParameters + 1] = g
 --   if (#gradParameters == 12) then gradParameters[#gradParameters] = gradParameters[#gradParameters] * 1.001 end
    -- print(g[3])
	    end
	end
    end
    print('end of this')

    local function storageInSet(set, storage, ifshow)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
        if not storageInSet(storages, storage, 0) then
    --          print(string.format("%depoch,storage added\n",k))
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
     --           print({storage[3]})
     --           print(storage:size())
              else
     --[[           print(string.format("%depoch,STORAGE NOT ADDED\n",k))
                print(storages)
                print({storage})
                print({storage[3]})
                print(torch.pointer(storage))
                print(nParameters)
                print(storage:size()) --]]
            end
        end
    --    print("\nnPars]:\n")
     --   print(nParameters) -- smth like 6169149
        

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage(), 0) --line30 for def of "storageInSet()", returns k-th entry in array (?)
            parameters[k]:set(flatStorage,   -- now parameters[k] points to flatStorage
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
      --    end
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
	
--[[	--------------------------------------------------------------------------- operative memory
 print({parameters})
 print({storages})
 print({nUsedParameters})
 print({cumSumOfHoles})
 print({maskParameters})
 print({flatStorage})
 print({flatParameters})
--]]
	
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage, 
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
--    print(string.format("\nnumpars= %d\n", #parameters))
 --   print(parameters)
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end




function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

return model_utils
