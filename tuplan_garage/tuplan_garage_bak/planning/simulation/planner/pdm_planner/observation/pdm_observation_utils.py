from typing import List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.maps_datatypes import (SemanticMapLayer,
                                                TrafficLightStatusData,
                                                TrafficLightStatusType) # BH가 추가함
from shapely.geometry import Polygon
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt # BH
from matplotlib.patches import Polygon as MplPolygon # BH
import numpy as np # BH

from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)

DRIVABLE_MAP_LAYERS = [
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.ROADBLOCK_CONNECTOR,
    SemanticMapLayer.CARPARK_AREA,
    SemanticMapLayer.CROSSWALK,
    SemanticMapLayer.STOP_LINE,
    SemanticMapLayer.WALKWAYS,
    SemanticMapLayer.LANE, # BH
    SemanticMapLayer.LANE_CONNECTOR, # BH
]


def get_drivable_area_map(
    map_api: AbstractMap,
    ego_state: EgoState,
    map_radius: float,
    # initialization,
    # current_input,
    # scenario_token
    ) -> PDMOccupancyMap:
    
    # query all drivable map elements around ego position
    position: Point2D = ego_state.center.point
    # map_radius = 600
    out_boundary = 100
    drivable_area = map_api.get_proximal_map_objects(position, map_radius, DRIVABLE_MAP_LAYERS)
    # BH, Roadblock visualization 추가
    # step0. DRIVEABLE AREA 뜯기
    
    
    # RBs = list(drivable_area.values())[0]
    # RB_cons = list(drivable_area.values())[1]
    # CAs = list(drivable_area.values())[2]
    # CWs = list(drivable_area.values())[3]
    # STs = list(drivable_area.values())[4]
    # WWs = list(drivable_area.values())[5]
    # LANEs = list(drivable_area.values())[6]
    # LANE_cons = list(drivable_area.values())[7]
    # route_roadblock_ids = initialization.route_roadblock_ids
    # traffic_light_status = [t for t in current_input.traffic_light_data]

    # print()
    # BH_RB_pack = {'id':[], 'polygon':[], 'con_id':[], 'con_polygon':[], 'route_id':[], 'current_iteration':0, 'token':"acb", 'goal':[]}
    # BH_LANE_pack = {'id':[], 'polygon':[], 'centerline':{'x':[], 'y':[]}, 'con_id':[], 'con_polygon':[], 'con_centerline':{'x':[], 'y':[]}, 'tr_green':{'x':[], 'y':[]}, 'tr_red':{'x':[], 'y':[]}, 'tr_yellow':{'x':[], 'y':[]}}
    # BH_CA_pack = {'id':[], 'polygon':[]}
    # BH_CW_pack = {'id':[], 'polygon':[]}
    # BH_ST_pack = {'id':[], 'polygon':[]}
    # BH_goal = {'route':route_roadblock_ids, 'goal':[initialization.mission_goal.point.x, initialization.mission_goal.point.y], 'ego_state':[ego_state.center.point.x, ego_state.center.point.y]}
    # BH_traffic_pack = {}
    # # Route_roadblock_id_seq = BH_visual_data["route_seq"]
    # # Goal_loc = BH_visual_data["goal_point"]
    # # ego_loc = BH_visual_data["ego_location"]
    # # cur_iteration = BH_visual_data["current_iteration"]
    # # current_scenario_token = BH_visual_data["scenario_token"]
    # if current_input.iteration.index != 0:
    #     raise ValueError
    # if current_input.iteration.index == 0:
    #     for tls in traffic_light_status:
    #         key = tls.lane_connector_id
    #         if tls.status == TrafficLightStatusType.GREEN:
    #             value = 'G'
    #         elif tls.status == TrafficLightStatusType.YELLOW:
    #             value = 'Y'
    #         elif tls.status == TrafficLightStatusType.RED:
    #             value = 'R'
    #         else:
    #             continue
    #         BH_traffic_pack[key] = value
        # for RB in RBs:
        #     BH_RB_pack['id'].append(RB.id)
        #     BH_RB_pack['polygon'].append(RB.polygon)
        # for RB_con in RB_cons:  
        #     BH_RB_pack['con_id'].append(RB_con.id)
        #     BH_RB_pack['con_polygon'].append(RB_con.polygon)
        # BH_RB_pack['route_id'] = initialization.route_roadblock_ids
        # BH_RB_pack['current_iteration'] = current_input.iteration.index
        # BH_RB_pack['token'] = scenario_token
        # BH_RB_pack['goal'] = [initialization.mission_goal.point.x, initialization.mission_goal.point.y]
        
    #     for LANE in LANEs:  
    #         BH_LANE_pack['id'].append(LANE.id)
    #         BH_LANE_pack['polygon'].append(LANE.polygon)
    #         disc_lane = LANE.baseline_path.discrete_path
    #         x = []
    #         y = []
    #         for point in disc_lane:
    #             BH_LANE_pack['centerline']['x'].append(point.x)
    #             BH_LANE_pack['centerline']['y'].append(point.y)
    #     for LANE_con in LANE_cons:  
    #         BH_LANE_pack['con_id'].append(LANE_con.id)
    #         BH_LANE_pack['con_polygon'].append(LANE_con.polygon)
    #         disc_lane = LANE_con.baseline_path.discrete_path
    #         x = []
    #         y = []
    #         if int(LANE_con.id) not in list(BH_traffic_pack.keys()):
    #             for point in disc_lane:
    #                 BH_LANE_pack['con_centerline']['x'].append(point.x)
    #                 BH_LANE_pack['con_centerline']['y'].append(point.y)
    #         else:
    #             if BH_traffic_pack[int(LANE_con.id)] == 'G':
    #                 for point in disc_lane:
    #                     BH_LANE_pack['tr_green']['x'].append(point.x)
    #                     BH_LANE_pack['tr_green']['y'].append(point.y)
    #             if BH_traffic_pack[int(LANE_con.id)] == 'Y':
    #                 for point in disc_lane:
    #                     BH_LANE_pack['tr_yellow']['x'].append(point.x)
    #                     BH_LANE_pack['tr_yellow']['y'].append(point.y)
    #             if BH_traffic_pack[int(LANE_con.id)] == 'R':
    #                 for point in disc_lane:
    #                     BH_LANE_pack['tr_red']['x'].append(point.x)
    #                     BH_LANE_pack['tr_red']['y'].append(point.y)
    #     for CA in CAs:  
    #         BH_CA_pack['id'].append(CA.id)
    #         BH_CA_pack['polygon'].append(CA.polygon)
    #     for CW in CWs:  
    #         BH_CW_pack['id'].append(CW.id)
    #         BH_CW_pack['polygon'].append(CW.polygon)
    #     for ST in STs:  
    #         BH_ST_pack['id'].append(ST.id)
    #         BH_ST_pack['polygon'].append(ST.polygon)
        
    # #     #step 3. visualization    
    #     plt.figure(figsize=(26,26))
    #     #step 3-1. plt 사이즈 고정(이쁘게 Visualization)
    #     figure_size_x = [BH_goal['ego_state'][0]-out_boundary, BH_goal['ego_state'][0]+out_boundary]
    #     figure_size_y = [BH_goal['ego_state'][1]-out_boundary, BH_goal['ego_state'][1]+out_boundary]
    #     plt.xlim(figure_size_x[0], figure_size_x[1])
    #     plt.ylim(figure_size_y[0], figure_size_y[1])
    #     # for con_polygon, con_id in zip(BH_RB_pack['con_polygon'], BH_RB_pack['con_id']):
    #     #     x, y = con_polygon.exterior.xy
    #     #     min_x, max_x, min_y, max_y = np.min(x)-out_boundary, np.max(x)+out_boundary, np.min(y)-out_boundary, np.max(y)+out_boundary
    #     #     figure_size_x[0] = min_x if figure_size_x[0] > min_x else figure_size_x[0]
    #     #     figure_size_x[1] = max_x if figure_size_x[1] < max_x else figure_size_x[1]
    #     #     figure_size_y[0] = min_y if figure_size_y[0] > min_y else figure_size_y[0]
    #     #     figure_size_y[1] = max_y if figure_size_y[1] < max_y else figure_size_y[1]
    #     # x_diff = figure_size_x[1]-figure_size_x[0]
    #     # y_diff = figure_size_y[1]-figure_size_y[0]
    #     # if x_diff > y_diff:
    #     #     plt.xlim(figure_size_x[0], figure_size_x[1])
    #     #     mean_y = np.mean(figure_size_y)
    #     #     plt.ylim(mean_y - x_diff/2, mean_y + x_diff/2)
    #     # else:
    #     #     plt.ylim(figure_size_y[0], figure_size_y[1])
    #     #     mean_x = np.mean(figure_size_x)
    #     #     plt.xlim(mean_x - y_diff/2, mean_x + y_diff/2)
    #     # Step3-2.진짜로 그리기
    #     mybox={'facecolor':'white','edgecolor':'r','boxstyle':'round','alpha':0.8}
    #     plt.plot(BH_goal['goal'][0], BH_goal['goal'][1], marker='*', markersize=22, color='green')
    #     plt.plot(BH_goal['ego_state'][0], BH_goal['ego_state'][1], marker='s', markersize=32, color='Red')
    #     # plt.scatter(BH_LANE_pack['centerline']['x'],  BH_LANE_pack['centerline']['y'], color = 'black', s=16.0, alpha=1.0)
    #     # plt.scatter(BH_LANE_pack['con_centerline']['x'],  BH_LANE_pack['con_centerline']['y'], color = 'gray', s=16.0, alpha=1.0)
    #     # plt.scatter(BH_LANE_pack['tr_green']['x'],  BH_LANE_pack['tr_green']['y'], color = 'green', s=24.0, alpha=1.0)
    #     # plt.scatter(BH_LANE_pack['tr_red']['x'],  BH_LANE_pack['tr_red']['y'], color = 'red', s=24.0, alpha=1.0)
    #     # plt.scatter(BH_LANE_pack['tr_yellow']['x'],  BH_LANE_pack['tr_yellow']['y'], color = 'yellow', s=24.0, alpha=1.0)
    #     for polygon in BH_RB_pack['polygon']:
    #         x, y = polygon.exterior.xy
    #         plt.fill(x, y, color='gray', edgecolor='black', alpha = 0.3)
    #     for polygon in BH_RB_pack['con_polygon']:
    #         x, y = polygon.exterior.xy
    #         plt.fill(x, y, color='gray', edgecolor='black', alpha = 0.3)
    #     for polygon in BH_ST_pack['polygon']:
    #         x, y = polygon.exterior.xy
    #         plt.fill(x, y, color='red', edgecolor='black', alpha = 0.9)
    #     for polygon in BH_CW_pack['polygon']:
    #         x, y = polygon.exterior.xy
    #         plt.fill(x, y, color='green', edgecolor='black', alpha = 0.9)
            
    # #     for con_polygon, con_id in zip(Roadblock_con_polygons, Roadblock_con_ids):
    # #         x, y = con_polygon.exterior.xy
    # #         if con_id in Route_roadblock_id_seq:
    # #             route_id_index = Route_roadblock_id_seq.index(con_id)
    # #             plt.fill(x, y, color='orange', edgecolor='blue')
    # #             plt.text(np.mean(x), np.mean(y), str(route_id_index), fontsize=14, ha='right', va='bottom', bbox=mybox)
    # #         else:
    # #             plt.fill(x, y, color='wheat', edgecolor='blue', alpha=0.4)
    # #     for lane_polygon, lane_id in zip(Lane_polygons, Lane_ids):
    # #         x, y = lane_polygon.exterior.xy
    # #         plt.fill(x, y, color='wheat', edgecolor='blue', alpha=0.4)
                
    #     plt.savefig(f'/home/workspace/tmp_save_fig/{scenario_token}_fig.png')
    
    
    
    # collect lane polygons in list, save on-route indices
    drivable_polygons: List[Polygon] = []
    drivable_polygon_ids: List[str] = []

    for type in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for roadblock in drivable_area[type]:
            for lane in roadblock.interior_edges:
                drivable_polygons.append(lane.polygon)
                drivable_polygon_ids.append(lane.id)

    for carpark in drivable_area[SemanticMapLayer.CARPARK_AREA]:
        drivable_polygons.append(carpark.polygon)
        drivable_polygon_ids.append(carpark.id)

    # create occupancy map with lane polygons
    drivable_area_map = PDMOccupancyMap(drivable_polygon_ids, drivable_polygons)

    return drivable_area_map# , BH_RB_pack
