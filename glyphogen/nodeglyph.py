from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:
    from .command_defs import CommandRepresentation

import numpy as np
import numpy.typing as npt

from glyphogen.hyperparameters import MAX_SEQUENCE_LENGTH


class Node:
    """A node, such as you would find in a vector design tool.

    It stores its position, and optional in/out handles as *absolute* coordinates.
    It knows where it is in a contour and can access its neighbors.
    Conversion to particular command and coordinate systems is handled elsewhere.
    """

    coordinates: npt.NDArray[np.float32]
    in_handle: Optional[npt.NDArray[np.float32]]
    out_handle: Optional[npt.NDArray[np.float32]]
    _contour: "NodeContour"
    ALIGNMENT_EPSILON = 0.1

    def __init__(self, coordinates, contour, in_handle=None, out_handle=None):
        self.coordinates = np.array(coordinates)
        self.in_handle = np.array(in_handle) if in_handle is not None else None
        self.out_handle = np.array(out_handle) if out_handle is not None else None
        self._contour = contour

    @property
    def index(self) -> int:
        return self._contour.nodes.index(self)

    @property
    def next(self) -> "Node":
        return self._contour.nodes[(self.index + 1) % len(self._contour.nodes)]

    @property
    def previous(self) -> "Node":
        return self._contour.nodes[self.index - 1]

    @property
    def is_line(self) -> bool:
        return self.in_handle is None and self.out_handle is None

    @property
    def is_horizontal_line(self) -> bool:
        return (
            self.is_line
            and self.index > 0
            and self.previous.coordinates[1] == self.coordinates[1]
        )

    @property
    def is_vertical_line(self) -> bool:
        return (
            self.is_line
            and self.index > 0
            and self.previous.coordinates[0] == self.coordinates[0]
        )

    @property
    def handles_horizontal(self) -> bool:
        return (
            self.in_handle is not None
            and self.out_handle is not None
            and abs(self.in_handle[1] - self.coordinates[1]) <= self.ALIGNMENT_EPSILON
            and abs(self.coordinates[1] - self.out_handle[1]) <= self.ALIGNMENT_EPSILON
        )

    @property
    def handles_vertical(self) -> bool:
        return (
            self.in_handle is not None
            and self.out_handle is not None
            and abs(self.in_handle[0] - self.coordinates[0]) <= self.ALIGNMENT_EPSILON
            and abs(self.coordinates[0] - self.out_handle[0]) <= self.ALIGNMENT_EPSILON
        )

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented

        coords_equal = np.allclose(self.coordinates, other.coordinates)

        in_handles_equal = (self.in_handle is None and other.in_handle is None) or (
            self.in_handle is not None
            and other.in_handle is not None
            and np.allclose(self.in_handle, other.in_handle)
        )

        out_handles_equal = (self.out_handle is None and other.out_handle is None) or (
            self.out_handle is not None
            and other.out_handle is not None
            and np.allclose(self.out_handle, other.out_handle)
        )

        if not (coords_equal and in_handles_equal and out_handles_equal):
            # print(f"Node mismatch:")
            # print(
            #     f"  Coords: {self.coordinates} vs {other.coordinates} (Equal: {coords_equal})"
            # )
            # print(
            #     f"  In Handles: {self.in_handle} vs {other.in_handle} (Equal: {in_handles_equal})"
            # )
            # print(
            #     f"  Out Handles: {self.out_handle} vs {other.out_handle} (Equal: {out_handles_equal})"
            # )
            return False

        return True


class NodeContour:
    nodes: List["Node"]

    def __init__(self, nodes: List["Node"]):
        self.nodes = nodes

    def normalize(self) -> None:
        if not self.nodes:
            return
        index_of_bottom_left = min(
            range(len(self.nodes)),
            key=lambda i: (self.nodes[i].coordinates[1], self.nodes[i].coordinates[0]),
        )
        self.nodes = (
            self.nodes[index_of_bottom_left:] + self.nodes[:index_of_bottom_left]
        )

    def __eq__(self, other):
        if not isinstance(other, NodeContour):
            return NotImplemented
        result = len(self.nodes) == len(other.nodes) and all(
            n1 == n2 for n1, n2 in zip(self.nodes, other.nodes)
        )
        return result

    def commands(
        self, vocabulary: type["CommandRepresentation"]
    ) -> Sequence["CommandRepresentation"]:
        return vocabulary.emit(self.nodes)

    def push(
        self,
        coordinates: npt.NDArray[np.float32],
        in_handle: Optional[npt.NDArray[np.float32]],
        out_handle: Optional[npt.NDArray[np.float32]],
    ) -> Node:
        node = Node(
            coordinates=coordinates,
            in_handle=in_handle,
            out_handle=out_handle,
            contour=self,
        )
        self.nodes.append(node)
        return node


class NodeGlyph:
    contours: List[NodeContour]
    origin: str

    def __init__(self, contours: List[NodeContour], origin="unknown"):
        self.contours = contours
        self.origin = origin

    def __eq__(self, other):
        if not isinstance(other, NodeGlyph):
            return NotImplemented
        result = len(self.contours) == len(other.contours) and all(
            c1 == c2 for c1, c2 in zip(self.contours, other.contours)
        )
        # Space for debugging code here
        return result

    def command_lists(
        self, vocabulary: type["CommandRepresentation"]
    ) -> List[Sequence["CommandRepresentation"]]:
        return [contour.commands(vocabulary) for contour in self.contours]

    @classmethod
    def from_command_lists(cls, contour_commands: List[List["CommandRepresentation"]]):
        contours = []
        representation_cls = (
            contour_commands[0][0].__class__
            if contour_commands and contour_commands[0]
            else None
        )
        if not representation_cls or not hasattr(
            representation_cls, "contour_from_commands"
        ):
            raise ValueError(
                "Commands must be of type CommandRepresentation to create NodeGlyph."
            )
        for cmds in contour_commands:
            contours.append(representation_cls.contour_from_commands(cmds))
        return cls(contours)

    def encode(
        self, vocabulary: type["CommandRepresentation"]
    ) -> Optional[List[npt.NDArray[np.float32]]]:
        contour_sequences = []

        for contour in self.contours:
            output: List[np.ndarray] = []

            def push_command(cmd: str, coords: List[float]):
                command_vector = np.zeros(vocabulary.command_width, dtype=np.float32)
                command_vector[vocabulary.encode_command(cmd)] = 1.0
                coord_array = np.array(coords, dtype=np.float32)
                padded_coords = np.pad(
                    coord_array, (0, vocabulary.coordinate_width - len(coords))  # type: ignore
                )
                output.append(np.concatenate((command_vector, padded_coords)))

            for command in contour.commands(vocabulary):
                push_command(command.command, command.coordinates)

            encoded_contour = np.array(output, dtype=np.float32)

            if encoded_contour.shape[0] > MAX_SEQUENCE_LENGTH:
                return None

            contour_sequences.append(encoded_contour)

        return contour_sequences if contour_sequences else None

    @classmethod
    def decode(
        cls, contour_sequences: List, representation_cls: type["CommandRepresentation"]
    ):
        """
        Decodes a list of ndarray sequences into a list of NodeCommand sequences.
        This is a stateless conversion from ndarray representation to command objects.
        """
        command_keys = list(representation_cls.grammar.keys())
        glyph_commands: List[List["CommandRepresentation"]] = []

        for ndarrays in contour_sequences:
            command_ndarray = ndarrays[:, : representation_cls.command_width]
            coord_ndarray = ndarrays[:, representation_cls.command_width :]

            contour_commands = []
            for i in range(command_ndarray.shape[0]):
                command_index = np.argmax(command_ndarray[i]).item()
                command_str = command_keys[command_index]

                num_coords = representation_cls.grammar[command_str]
                coords_slice = coord_ndarray[i, :num_coords]
                contour_commands.append(
                    representation_cls(command_str, coords_slice.tolist())
                )
                if command_str == "EOS":
                    break

            glyph_commands.append(contour_commands)

        return cls.from_command_lists(glyph_commands)

    def to_debug_string(self):
        path_data: List[str] = []
        for contour in self.contours:
            for node in contour.nodes:
                path_data.append(f"N {node.coordinates[0]} {node.coordinates[1]}")
                if node.in_handle is not None:
                    path_data.append(f"IN {node.in_handle[0]} {node.in_handle[1]}")
                if node.out_handle is not None:
                    path_data.append(f"OUT {node.out_handle[0]} {node.out_handle[1]}")
            path_data.append("Z")
        return " ".join(path_data)

    def normalize(self):
        for contour in self.contours:
            contour.normalize()
        self.contours.sort(
            key=lambda c: (
                (c.nodes[0].coordinates[1], c.nodes[0].coordinates[0])
                if c.nodes
                else (float("inf"), float("inf"))
            )
        )
        return self
