using Opencraft.Terrain;
using Opencraft.Terrain.Authoring;
using Opencraft.Terrain.Blocks;
using Opencraft.Terrain.Utilities;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using Unity.Collections;
using UnityEngine;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Unity.Transforms;

[assembly: RegisterGenericJobType(typeof(SortJob<int2, Int2DistanceComparer>))]
namespace Opencraft.Terrain
{
    [WorldSystemFilter(WorldSystemFilterFlags.ServerSimulation)]
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    [UpdateAfter(typeof(TerrainStructuresSystem))]
    [BurstCompile]

    public partial struct TerrainLogicSystem : ISystem
    {
        private double tickRate;
        private float timer;
        private BufferLookup<BlockLogicState> _terrainLogicStateLookup;
        private BufferLookup<BlockDirection> _terrainDirectionLookup;
        private BufferLookup<TerrainBlocks> _terrainBlocksLookup;
        private BufferLookup<UpdatedBlocks> _terrainUpdatedLookup;
        private BufferLookup<InputBlocks> _terrainInputLookup;
        private BufferLookup<GateBlocks> _terrainGateLookup;
        private BufferLookup<ActiveGateBlocks> _terrainActiveGateLookup;
        private ComponentLookup<TerrainNeighbors> _terrainNeighborsLookup;
        private ComponentLookup<TerrainArea> _terrainAreaLookup;
        private NativeArray<Entity> terrainAreasEntities;

        public struct LogicBlockData
        {
            public int3 BlockLocation;
            public Entity TerrainEntity;
        }
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<TerrainArea>();
            tickRate = 1;
            timer = 0;
            _terrainLogicStateLookup = state.GetBufferLookup<BlockLogicState>(isReadOnly: false);
            _terrainDirectionLookup = state.GetBufferLookup<BlockDirection>(isReadOnly: false);
            _terrainBlocksLookup = state.GetBufferLookup<TerrainBlocks>(isReadOnly: false);
            _terrainUpdatedLookup = state.GetBufferLookup<UpdatedBlocks>(isReadOnly: false);
            _terrainInputLookup = state.GetBufferLookup<InputBlocks>(isReadOnly: false);
            _terrainGateLookup = state.GetBufferLookup<GateBlocks>(isReadOnly: false);
            _terrainActiveGateLookup = state.GetBufferLookup<ActiveGateBlocks>(isReadOnly: false);
            _terrainNeighborsLookup = state.GetComponentLookup<TerrainNeighbors>(isReadOnly: false);
            _terrainAreaLookup = state.GetComponentLookup<TerrainArea>(isReadOnly: false);
        }

        public void OnDestroy(ref SystemState state)
        {
        }

        public void OnUpdate(ref SystemState state)
        {
            if (timer < tickRate)
            {
                timer += Time.deltaTime;
                return;
            }
            timer = 0;
            _terrainLogicStateLookup.Update(ref state);
            _terrainDirectionLookup.Update(ref state);
            _terrainBlocksLookup.Update(ref state);
            _terrainUpdatedLookup.Update(ref state);
            _terrainInputLookup.Update(ref state);
            _terrainGateLookup.Update(ref state);
            _terrainActiveGateLookup.Update(ref state);
            _terrainNeighborsLookup.Update(ref state);
            _terrainAreaLookup.Update(ref state);

            var terrainAreasQuery = SystemAPI.QueryBuilder().WithAll<TerrainArea, LocalTransform>().Build();
            terrainAreasEntities = terrainAreasQuery.ToEntityArray(state.WorldUpdateAllocator);

            //EntityCommandBuffer ecb = new EntityCommandBuffer(Allocator.TempJob);
            //EntityCommandBuffer.ParallelWriter parallelEcb = ecb.AsParallelWriter();

            JobHandle populateHandle = new UpdateTerrainLogic
            {
                terrainAreasEntities = terrainAreasEntities,
                //ecb = parallelEcb,
                terrainLogicStateLookup = _terrainLogicStateLookup,
                terrainDirectionLookup = _terrainDirectionLookup,
                terrainBlocksLookup = _terrainBlocksLookup,
                terrainUpdatedLookup = _terrainUpdatedLookup,
                terrainInputLookup = _terrainInputLookup,
                terrainGateLookup = _terrainGateLookup,
                terrainActiveGateLookup = _terrainActiveGateLookup,
                terrainNeighborsLookup = _terrainNeighborsLookup,
                terrainAreaLookup = _terrainAreaLookup
            }.Schedule(terrainAreasEntities.Length, 1);

            populateHandle.Complete();
            terrainAreasEntities.Dispose();
        }

        [BurstCompile]
        partial struct UpdateTerrainLogic : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Entity> terrainAreasEntities;

            //public EntityCommandBuffer.ParallelWriter ecb;
            [NativeDisableParallelForRestriction] public BufferLookup<BlockLogicState> terrainLogicStateLookup;
            [NativeDisableParallelForRestriction] public BufferLookup<BlockDirection> terrainDirectionLookup;
            [NativeDisableParallelForRestriction] public BufferLookup<TerrainBlocks> terrainBlocksLookup;
            [NativeDisableParallelForRestriction] public BufferLookup<UpdatedBlocks> terrainUpdatedLookup;
            [NativeDisableParallelForRestriction] public BufferLookup<InputBlocks> terrainInputLookup;
            [NativeDisableParallelForRestriction] public BufferLookup<GateBlocks> terrainGateLookup;
            [NativeDisableParallelForRestriction] public BufferLookup<ActiveGateBlocks> terrainActiveGateLookup;
            [NativeDisableParallelForRestriction] public ComponentLookup<TerrainNeighbors> terrainNeighborsLookup;
            [NativeDisableParallelForRestriction] public ComponentLookup<TerrainArea> terrainAreaLookup;


            public void Execute(int jobIndex)
            {
                Entity terrainEntity = terrainAreasEntities[jobIndex];

                DynamicBuffer<int3> updateBlocks = terrainUpdatedLookup[terrainEntity].Reinterpret<int3>();
                if (updateBlocks.Length == 0) return;
                NativeArray<int3> updateBlocksCopy = updateBlocks.ToNativeArray(Allocator.Temp);
                updateBlocks.Clear();

                DynamicBuffer<BlockType> blockTypeBuffer = terrainBlocksLookup[terrainEntity].Reinterpret<BlockType>();
                DynamicBuffer<InputBlocks> inputBlocks = terrainInputLookup[terrainEntity];
                DynamicBuffer<GateBlocks> gateBlocks = terrainGateLookup[terrainEntity];
                DynamicBuffer<ActiveGateBlocks> activeGateBlocks = terrainActiveGateLookup[terrainEntity];

                NativeList<int3> toReevaluate = new NativeList<int3>(4096, Allocator.Temp);

                for (int i = 0; i < updateBlocksCopy.Length; i++)
                {
                    int3 blockLoc = updateBlocksCopy[i];
                    int blockIndex = TerrainUtilities.BlockLocationToIndex(ref blockLoc);
                    BlockType blockType = blockTypeBuffer[blockIndex];

                    toReevaluate.Add(new int3(blockLoc));

                    if (blockType == BlockType.Air)
                    {
                        RemoveInputBlock(blockLoc, ref inputBlocks);
                        RemoveGateBlock(blockLoc, ref gateBlocks);
                        RemoveActiveGateBlock(blockLoc, ref activeGateBlocks);
                    }
                    else if (BlockData.IsInput(blockType) || blockType == BlockType.NOT_Gate)
                        inputBlocks.Add(new InputBlocks { blockLoc = blockLoc });
                    else if (BlockData.IsGate(blockType))
                        gateBlocks.Add(new GateBlocks { blockLoc = blockLoc });
                }
                updateBlocksCopy.Dispose();
                if (toReevaluate.Length > 0)
                {
                    PropagateLogicState(ref toReevaluate, ref terrainEntity, false);
                }
                toReevaluate.Dispose();
                NativeList<int3> inputCoords = new NativeList<int3>(Allocator.Temp);
                for (int i = 0; i < inputBlocks.Length; i++)
                {
                    inputCoords.Add(inputBlocks[i].blockLoc);
                }
                PropagateLogicState(ref inputCoords, ref terrainEntity, true);
                inputCoords.Dispose();

                NativeList<int3> gateCoords = new NativeList<int3>(Allocator.Temp);
                for (int i = 0; i < gateBlocks.Length; i++)
                {
                    gateCoords.Add(gateBlocks[i].blockLoc);
                }
                CheckGateState(gateCoords, ref terrainEntity);
                gateCoords.Dispose();
            }

            private void PropagateLogicState(ref NativeList<int3> logicBlocks, ref Entity terrainEntity, bool inputLogicState)
            {
                NativeQueue<(int3, Entity, bool)> logicQueue = new NativeQueue<(int3, Entity, bool)>(Allocator.Temp);
                foreach (int3 blockLoc in logicBlocks)
                {
                    logicQueue.Enqueue((new int3(blockLoc), terrainEntity, inputLogicState));
                }
                while (logicQueue.Count > 0)
                {
                    logicQueue.TryDequeue(out (int3, Entity, bool) entry);
                    int3 blockLoc = entry.Item1;
                    Entity blockEntity = entry.Item2;
                    bool logicState = entry.Item3;

                    int blockIndex = TerrainUtilities.BlockLocationToIndex(ref blockLoc);
                    BlockType currentBlockType = terrainBlocksLookup[blockEntity].Reinterpret<BlockType>()[blockIndex];
                    Direction currentOutputDirection = terrainDirectionLookup[blockEntity].Reinterpret<Direction>()[blockIndex];

                    if (logicState && (currentBlockType == BlockType.Off_Input)) continue;

                    TerrainNeighbors neighbors = terrainNeighborsLookup[blockEntity];
                    Entity neighborXN = neighbors.neighborXN;
                    Entity neighborXP = neighbors.neighborXP;
                    Entity neighborZN = neighbors.neighborZN;
                    Entity neighborZP = neighbors.neighborZP;
                    NativeList<Entity> terrainEntities = new NativeList<Entity>(Allocator.Temp) { blockEntity, neighborXN, neighborXP, neighborZN, neighborZP };

                    if (currentBlockType == BlockType.Clock)
                    {
                        DynamicBuffer<BlockLogicState> blockLogicStates = terrainLogicStateLookup[blockEntity];
                        DynamicBuffer<bool> boolLogicStates = blockLogicStates.Reinterpret<bool>();
                        boolLogicStates[blockIndex] = !boolLogicStates[blockIndex];
                        logicState = boolLogicStates[blockIndex];
                    }

                    if (BlockData.IsTwoInputGate(currentBlockType))
                    {
                        EvaluateNeighbour(currentOutputDirection, blockLoc, ref terrainEntities, logicState, ref logicQueue);
                        continue;
                    }

                    if (currentBlockType == BlockType.NOT_Gate)
                    {
                        Direction inputDirection = BlockData.OppositeDirections[(int)currentOutputDirection];
                        int3 notNormalisedBlockLoc = (blockLoc + BlockData.Int3Directions[(int)inputDirection]);
                        int terrainEntityIndex = GetOffsetIndex(notNormalisedBlockLoc);
                        Entity neighborEntity = terrainEntities[terrainEntityIndex];
                        if (neighborEntity == Entity.Null) continue;
                        int lowestCoord = math.min(notNormalisedBlockLoc.x, notNormalisedBlockLoc.z);
                        int num_sixteens = lowestCoord / 16 + 1;
                        int3 actualBlockLoc = (notNormalisedBlockLoc + new int3(16, 0, 16) * num_sixteens) % 16;
                        int blockIndex2 = TerrainUtilities.BlockLocationToIndex(ref actualBlockLoc);
                        DynamicBuffer<BlockLogicState> blockLogicStates = terrainLogicStateLookup[neighborEntity];
                        DynamicBuffer<bool> boolLogicStates = blockLogicStates.Reinterpret<bool>();
                        bool NOTInputState = boolLogicStates[blockIndex2];

                        EvaluateNeighbour(currentOutputDirection, blockLoc, ref terrainEntities, !NOTInputState, ref logicQueue);
                        continue;
                    }

                    Direction[] allDirections = BlockData.AllDirections;
                    for (int i = 0; i < allDirections.Length; i++)
                    {
                        Direction outputDirection = allDirections[i];
                        EvaluateNeighbour(outputDirection, blockLoc, ref terrainEntities, logicState, ref logicQueue);
                    }
                    terrainEntities.Dispose();
                }
                logicQueue.Dispose();
            }

            private void CheckGateState(NativeList<int3> gateBlocks, ref Entity blockEntity)
            {
                NativeQueue<int3> gateQueue = new NativeQueue<int3>(Allocator.Temp);
                foreach (int3 blockLoc in gateBlocks)
                {
                    gateQueue.Enqueue(blockLoc);
                }
                while (gateQueue.Count > 0)
                {
                    gateQueue.TryDequeue(out int3 blockLoc);

                    int blockIndex = TerrainUtilities.BlockLocationToIndex(ref blockLoc);
                    TerrainArea terrainArea = terrainAreaLookup[blockEntity];
                    int3 globalPos = terrainArea.location * Env.AREA_SIZE + blockLoc;
                    BlockType currentBlockType = terrainBlocksLookup[blockEntity].Reinterpret<BlockType>()[blockIndex];
                    DynamicBuffer<BlockLogicState> blockLogicState = terrainLogicStateLookup[blockEntity];
                    DynamicBuffer<Direction> directionStates = terrainDirectionLookup[blockEntity].Reinterpret<Direction>();
                    Direction currentDirection = directionStates[blockIndex];
                    DynamicBuffer<bool> boolLogicState = blockLogicState.Reinterpret<bool>();

                    NativeList<Direction> inputDirections = new NativeList<Direction>(Allocator.Temp) { };
                    GetInputDirections(ref inputDirections, currentDirection);
                    int requiredInputs = 0;
                    switch (currentBlockType)
                    {
                        case BlockType.AND_Gate:
                            requiredInputs = 2;
                            break;
                        case BlockType.OR_Gate:
                        case BlockType.XOR_Gate:
                            requiredInputs = 1;
                            break;
                        default:
                            break;
                    }

                    TerrainNeighbors neighbors = terrainNeighborsLookup[blockEntity];
                    Entity neighborXN = neighbors.neighborXN;
                    Entity neighborXP = neighbors.neighborXP;
                    Entity neighborZN = neighbors.neighborZN;
                    Entity neighborZP = neighbors.neighborZP;
                    NativeList<Entity> terrainEntities = new NativeList<Entity> { blockEntity, neighborXN, neighborXP, neighborZN, neighborZP };

                    int3[] directions = BlockData.Int3Directions;
                    int onCount = 0;
                    for (int i = 0; i < inputDirections.Length; i++)
                    {
                        int3 notNormalisedBlockLoc = (blockLoc + directions[(int)inputDirections[i]]);
                        int terrainEntityIndex = GetOffsetIndex(notNormalisedBlockLoc);
                        Entity neighborEntity = terrainEntities[terrainEntityIndex];
                        if (neighborEntity == Entity.Null) continue;

                        int lowestCoord = math.min(notNormalisedBlockLoc.x, notNormalisedBlockLoc.z);
                        int num_sixteens = lowestCoord / 16 + 1;
                        int3 actualBlockLoc = (notNormalisedBlockLoc + new int3(16, 0, 16) * num_sixteens) % 16;
                        int blockIndex2 = TerrainUtilities.BlockLocationToIndex(ref actualBlockLoc);

                        DynamicBuffer<TerrainBlocks> terrainBlocks = terrainBlocksLookup[neighborEntity];
                        DynamicBuffer<BlockType> blockTypes = terrainBlocks.Reinterpret<BlockType>();
                        DynamicBuffer<BlockLogicState> blockLogicStates2 = terrainLogicStateLookup[neighborEntity];
                        DynamicBuffer<bool> boolLogicStates2 = blockLogicStates2.Reinterpret<bool>();
                        BlockType currentBlock = blockTypes[blockIndex2];

                        if (boolLogicStates2[blockIndex2])
                        {
                            onCount++;
                        }
                    }

                    DynamicBuffer<ActiveGateBlocks> activeGateBlocks = terrainActiveGateLookup[blockEntity];
                    DynamicBuffer<UpdatedBlocks> updatedBlocks = terrainUpdatedLookup[blockEntity];

                    if ((onCount >= requiredInputs && (currentBlockType == BlockType.AND_Gate || currentBlockType == BlockType.OR_Gate)) || (onCount == requiredInputs && currentBlockType == BlockType.XOR_Gate))
                    {
                        boolLogicState[blockIndex] = true;
                        activeGateBlocks.Add(new ActiveGateBlocks { blockLoc = blockLoc });
                    }
                    else
                    {
                        boolLogicState[blockIndex] = false;
                        RemoveActiveGateBlock(blockLoc, ref activeGateBlocks);
                        updatedBlocks.Add(new UpdatedBlocks { blockLoc = blockLoc });
                    }

                    terrainEntities.Dispose();
                    inputDirections.Dispose();
                }
            }

            private void EvaluateNeighbour(Direction outputDirection, int3 blockLoc, ref NativeList<Entity> terrainEntities, bool logicState, ref NativeQueue<(int3, Entity, bool)> logicQueue)
            {
                int3 direction = BlockData.Int3Directions[(int)outputDirection];
                int3 notNormalisedBlockLoc = (blockLoc + direction);
                int terrainEntityIndex = GetOffsetIndex(notNormalisedBlockLoc);
                Entity neighborEntity = terrainEntities[terrainEntityIndex];
                if (neighborEntity == Entity.Null) return;

                int lowestCoord = math.min(notNormalisedBlockLoc.x, notNormalisedBlockLoc.z);
                int num_sixteens = lowestCoord / 16 + 1;
                int3 actualBlockLoc = (notNormalisedBlockLoc + new int3(16, 0, 16) * num_sixteens) % 16;
                int blockIndex = TerrainUtilities.BlockLocationToIndex(ref actualBlockLoc);

                DynamicBuffer<TerrainBlocks> terrainBlocks = terrainBlocksLookup[neighborEntity];
                DynamicBuffer<BlockType> blockTypes = terrainBlocks.Reinterpret<BlockType>();
                DynamicBuffer<BlockLogicState> blockLogicState = terrainLogicStateLookup[neighborEntity];
                DynamicBuffer<bool> boolLogicState = blockLogicState.Reinterpret<bool>();
                BlockType currentBlock = blockTypes[blockIndex];
                int currentBlockIndex = (int)currentBlock;

                if (BlockData.CanReceiveLogic[currentBlockIndex])
                {
                    if (boolLogicState[blockIndex] != logicState)
                    {
                        boolLogicState[blockIndex] = logicState;
                        if (currentBlock == BlockType.Off_Wire || currentBlock == BlockType.On_Wire || currentBlock == BlockType.On_Lamp)
                        {
                            logicQueue.Enqueue((blockLoc, neighborEntity, logicState));
                        }
                        if (logicState) blockTypes[blockIndex] = (BlockData.OnState[currentBlockIndex]);
                        else blockTypes[blockIndex] = (BlockData.OffState[currentBlockIndex]);
                    }
                }
            }

            private int GetOffsetIndex(int3 blockLoc)
            {
                switch (blockLoc.x)
                {
                    case -1: return 1;
                    case 16: return 2;
                    default: break;
                }
                switch (blockLoc.z)
                {
                    case -1: return 3;
                    case 16: return 4;
                    default: break;
                }
                return 0;
            }

            private void GetInputDirections(ref NativeList<Direction> inputDirections, Direction currentDirection)
            {
                switch (currentDirection)
                {
                    case Direction.XN:
                    case Direction.XP:
                        inputDirections.Add(Direction.ZN);
                        inputDirections.Add(Direction.ZP);
                        break;
                    case Direction.ZN:
                    case Direction.ZP:
                        inputDirections.Add(Direction.XN);
                        inputDirections.Add(Direction.XP);
                        break;
                    default:
                        break;
                }
            }


            private void RemoveInputBlock(int3 blockLoc, ref DynamicBuffer<InputBlocks> inputBlocks)
            {
                for (int i = 0; i < inputBlocks.Length; i++)
                {
                    if (inputBlocks[i].blockLoc.Equals(blockLoc))
                    {
                        inputBlocks.RemoveAt(i);
                        return;
                    }
                }
            }

            private void RemoveGateBlock(int3 blockLoc, ref DynamicBuffer<GateBlocks> gateBlocks)
            {
                for (int i = 0; i < gateBlocks.Length; i++)
                {
                    if (gateBlocks[i].blockLoc.Equals(blockLoc))
                    {
                        gateBlocks.RemoveAt(i);
                        return;
                    }
                }
            }

            private void RemoveActiveGateBlock(int3 blockLoc, ref DynamicBuffer<ActiveGateBlocks> activeGateBlocks)
            {
                for (int i = 0; i < activeGateBlocks.Length; i++)
                {
                    if (activeGateBlocks[i].blockLoc.Equals(blockLoc))
                    {
                        activeGateBlocks.RemoveAt(i);
                        return;
                    }
                }
            }

        }
    }
}