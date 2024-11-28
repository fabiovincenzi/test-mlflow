import re
import sys

import git
from string_utils import camel_to_snake, snake_to_pascal


def get_git_root():
    repo = git.Repo(".", search_parent_directories=True)
    return repo.working_tree_dir + "/"


INDENT = " " * 4
INDENT2 = INDENT * 2
SCHEMA_EXTENSION_MODULE = "mlflow.server.graphql.graphql_schema_extensions"
SCHEMA_EXTENSION = get_git_root() + "mlflow/server/graphql/graphql_schema_extensions.py"
AUTOGENERATED_SCHEMA = get_git_root() + "mlflow/server/graphql/autogenerated_graphql_schema.py"
AUTOGENERATED_SDL_SCHEMA = get_git_root() + "mlflow/server/js/src/graphql/autogenerated_schema.gql"
DUMMY_FIELD = (
    "dummy = graphene.Boolean(description="
    "'Dummy field required because GraphQL does not support empty types.')"
)


def get_package_name(method_descriptor):
    return method_descriptor.containing_service.file.package


# Get method name in snake case. Result is package name followed by the method name.
def get_method_name(method_descriptor):
    return get_package_name(method_descriptor) + "_" + camel_to_snake(method_descriptor.name)


def get_descriptor_full_pascal_name(field_descriptor):
    return snake_to_pascal(field_descriptor.full_name.replace(".", "_"))


def method_descriptor_to_generated_pb2_file_name(method_descriptor):
    return re.sub(r"\.proto", "_pb2", method_descriptor.containing_service.file.name)


def debugLog(log):
    print(log, file=sys.stderr)