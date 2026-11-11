# Public, generic ROS/Gazebo telemetry probe. No private ASTRA source copied here.
FROM osrf/ros:noetic-desktop-full-focal
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive ROS_MASTER_URI=http://127.0.0.1:11311 ROS_HOSTNAME=127.0.0.1 QT_QPA_PLATFORM=offscreen GAZEBO_MODEL_DATABASE_URI=
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates libeigen3-dev && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/OnOrbitROS/Simulation.git /tmp/onorbitros && cd /tmp/onorbitros \
 && git checkout --detach c2edc7414cc90a6f0421c16702ef44c5895fd151 \
 && test "$(git rev-parse HEAD)" = 'c2edc7414cc90a6f0421c16702ef44c5895fd151' \
 && mkdir -p /catkin_ws/src /usr/share/onorbitros \
 && cp -a orbit_publisher_pkg /catkin_ws/src/ \
 && cp LICENSE /usr/share/onorbitros/LICENSE \
 && rm -rf /tmp/onorbitros
# Upstream package metadata has an undeclared/unused orbit_msgs Catkin dependency;
# remove dependency metadata only, keep C++ source unchanged.
RUN sed -i '/<build_depend>orbit_msgs<\/build_depend>/d; /<build_export_depend>orbit_msgs<\/build_export_depend>/d' /catkin_ws/src/orbit_publisher_pkg/package.xml \
 && sed -i 's/CATKIN_DEPENDS roscpp std_msgs geometry_msgs orbit_msgs/CATKIN_DEPENDS roscpp std_msgs geometry_msgs/' /catkin_ws/src/orbit_publisher_pkg/CMakeLists.txt \
 && source /opt/ros/noetic/setup.bash \
 && cd /catkin_ws && catkin_make -j1 -DCMAKE_BUILD_TYPE=Release
COPY start.sh /usr/local/bin/onorbit-start.sh
COPY api.py /opt/onorbitros-public-probe/api.py
RUN bash -n /usr/local/bin/onorbit-start.sh && python3 -m py_compile /opt/onorbitros-public-probe/api.py
EXPOSE 10000
CMD ["/bin/bash", "/usr/local/bin/onorbit-start.sh"]
